#include <metal_stdlib>
using namespace metal;
float fluidDistance(float3 p, device const float *field, CupBody body);
float4 fluidSample(float3 p, device const float *field);
float3 fluidGradient(float3 p, device const float *field);
// Scalars deliberately avoid Swift/MSL float3 alignment differences.
// u: width height time level yaw pitch roll tiltX tiltY heat ambient wind,
// cupRGB liquidRGB cremaRGB saucerRGB steamRGB, retro pour dotsCount dither weather,
// real surfaceHeight cinematic fluidTime; the host appends exposure as u[36].
float hash21(float2 p) { return fract(sin(dot(p,float2(127.1,311.7)))*43758.5453); }
float noise(float2 p) {
    float2 i=floor(p), f=fract(p); f=f*f*(3.-2.*f);
    return mix(mix(hash21(i),hash21(i+float2(1,0)),f.x),
               mix(hash21(i+float2(0,1)),hash21(i+1.),f.x),f.y);
}
// Integer hash: stable for large lattice coordinates, unlike sin().
uint hashU(uint3 q) {
    uint h=q.x*1597334677u^q.y*3812015801u^q.z*2798796415u;
    h=(h^(h>>16))*2246822519u;h^=h>>13;h*=3266489917u;return h^(h>>16);
}
// y wraps every 256 lattice cells so scrolling steam can wrap its clock seamlessly.
float hash31(float3 p) {int3 i=int3(p);return float(hashU(uint3(i.x,i.y&255,i.z)))*(1./4294967296.);}
float2 hash22(float2 p) {int2 i=int2(p);uint h=hashU(uint3(i.x,i.y,77));return float2(h&65535u,h>>16)*(1./65536.);}
float noise3(float3 p) {
    float3 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
    float a=mix(hash31(i),hash31(i+float3(1,0,0)),f.x),b=mix(hash31(i+float3(0,1,0)),hash31(i+float3(1,1,0)),f.x);
    float c=mix(hash31(i+float3(0,0,1)),hash31(i+float3(1,0,1)),f.x),d=mix(hash31(i+float3(0,1,1)),hash31(i+1.),f.x);
    return mix(mix(a,b,f.y),mix(c,d,f.y),f.z);
}
float wave(float2 p, device const float *w) {
    float2 g=clamp((p/.72+1.)*.5*float2(53,25),float2(0),float2(52.999,24.999));
    int2 i=int2(g); float2 f=fract(g); int k=i.y*54+i.x;
    return mix(mix(w[k],w[k+1],f.x),mix(w[k+54],w[k+55],f.x),f.y);
}
float2 slopes(device const float *u) {
    float c=cos(u[4]),s=sin(u[4]);
    return float2(u[7]*c+u[8]*s,u[8]*c-u[7]*s)*.45;
}
float liquidHeight(float2 p, device const float *u, device const float *w, CupBody body) {
    float r=length(p);
    float meniscus=.014*exp(-max(0.f,.73-r)*48.);
    return min(.988f,.30+.62*u[3]+dot(p,slopes(u))+clamp(wave(p,w)*.045,-.035f,.035f)+meniscus);
}
// Inner glaze radius of the resting cup at height y (local frame).
float cupInnerRadius(float y) {
    return y<.27 ? .66+.09*(y-.13)/.14-.057 : .75+.05*(min(y,1.f)-.27)/.73-.048;
}
// Render-only props: the fluid and cup never collide with them.
constant float3 SPOON_AT=float3(-.15,-.048,.80);
constant float2 SPOON_DIR=float2(.831,.556);
float spoonShape(float3 p) {
    float3 q=p-SPOON_AT;
    float s=dot(q.xz,SPOON_DIR),w=dot(q.xz,float2(-SPOON_DIR.y,SPOON_DIR.x));
    // Bowl: a 1 mm ellipsoidal shell, open above its rim plane.
    float3 b=float3(s,q.y-.045,w),R=float3(.19,.045,.12);
    float k0=length(b/R),k1=length(b/(R*R));
    float bowl=max(abs(k0*(k0-1.)/k1)-.005,b.y);
    // Flattened handle rising from the bowl to rest on the saucer rim.
    float hs=clamp(s,.17f,1.22f);
    float halfW=mix(.011f,.026f,smoothstep(.3f,1.1f,hs)),halfT=.0055;
    float2 c=float2(w,q.y-.045-(hs-.17)*.096);
    float cross=(length(c/float2(halfW,halfT))-1.)*halfT;
    float handle=length(float2(max(cross,0.f),s-hs))+min(cross,0.f);
    return min(bowl,handle);
}
constant float3 PACKET_AT=float3(-1.08,-.13,1.58);
float2 packetLocal(float3 p) {
    float3 q=p-PACKET_AT;float c=cos(.45),s=sin(.45);
    return float2(c*q.x+s*q.z,-s*q.x+c*q.z);
}
float packetShape(float3 p) {
    float2 xz=packetLocal(p);float y=p.y-PACKET_AT.y-.011;
    float3 d=abs(float3(xz.x,y,xz.y))-float3(.26,.008,.18);
    float pillow=.003*max(0.f,1.-xz.x*xz.x/.06)*max(0.f,1.-xz.y*xz.y/.03);
    return length(max(d,0.f))+min(max(d.x,max(d.y,d.z)),0.f)-.003-pillow;
}
float2 props(float3 p) {
    float s=spoonShape(p),k=packetShape(p);
    return s<k?float2(s,5):float2(k,6);
}
float occluder(float3 p,CupBody body) {return min(crockery(p,body).x,props(p).x);}
float2 scene(float3 p, device const float *u, device const float *w, CupBody body) {
    float2 hit=crockery(p,body);
    // The legacy disc stays inside the glaze; past the lower wall it showed as an amber band.
    float liquid=u[32]>.5 ? fluidDistance(p,w,body) :
        max(max((p.y-liquidHeight(p.xz,u,w,body))*.7,length(p.xz)-min(.745f,cupInnerRadius(p.y))),.125-p.y);
    if(liquid<hit.x) hit=float2(liquid,2);
    if(p.y+.13<hit.x) hit=float2(p.y+.13,3);
    float2 prop=props(p);
    if(prop.x<hit.x) hit=prop;
    // A narrow real pour, enabled only while the existing simulation refills.
    if(u[32]<.5 && u[28]>.5 && p.y>.92 && p.y<2.7) {
        float stream=length(p.xz-float2(-.15+.007*sin(p.y*15.+u[2]*8.),-.18))-.018;
        if(stream<hit.x) hit=float2(stream,2);
    }
    return hit;
}
float3 normalAt(float3 p, device const float *u, device const float *w, CupBody body) {
    float e=.0015;
    return normalize(float3(scene(p+float3(e,0,0),u,w,body).x-scene(p-float3(e,0,0),u,w,body).x,
                            scene(p+float3(0,e,0),u,w,body).x-scene(p-float3(0,e,0),u,w,body).x,
                            scene(p+float3(0,0,e),u,w,body).x-scene(p-float3(0,0,e),u,w,body).x));
}
constant float3 RED_DIR=float3(-.8589,.4581,.2290);
constant float3 TEAL_DIR=float3(.2129,.4790,-.8516);
// Background lights are seen through a defocused lens: each is a disc of the
// aperture's angular size, so the blur is analytic and stable under TAA.
constant float BACKDROP_BLUR=.062;
constant float TABLE_HALF=6.;
float headlightPhase(device const float *u) {return fmod(u[2]+4.,22.);}
float headlightTravel(device const float *u) {return mix(-4.f,4.f,headlightPhase(u)/6.);}
float3 headlightLamp(device const float *u,float side) {return float3(headlightTravel(u)*3.+side,4.2,-12.);}
float headlightFade(device const float *u) {
    float phase=headlightPhase(u);
    return phase<6. ? smoothstep(0.f,.9f,phase)*(1.-smoothstep(4.8f,6.f,phase)) : 0.;
}
float wrapAngle(float a) {return a-6.2831853*floor(a/6.2831853+.5);}
float sdRoundRect(float2 p,float2 b,float r) {
    float2 d=abs(p)-b+r;return length(max(d,0.f))+min(max(d.x,d.y),0.f)-r;
}
// Defocused line light: a thin tube spread over the bokeh radius r.
float neonTube(float d,float r) {return (1.-smoothstep(0.f,r+.004f,d))*.012/(r+.012);}
float3 backdrop(float3 d,float blur,float3 origin,device const float *u) {
    float el=asin(clamp(d.y,-1.f,1.f)),az=atan2(d.z,d.x),cs=cos(el),r=blur*.5+.002;
    float3 col=mix(float3(.009,.007,.006),float3(.026,.021,.018),smoothstep(-.24f,-.02f,el));
    col=mix(col,float3(.040,.033,.028),smoothstep(.15f,.7f,el));
    // Back-lit menu board on the wall behind the default camera: the soft
    // vertical highlight that tells glaze and coffee they are glossy.
    float2 mq=float2(wrapAngle(az-1.)*cs,el-.22);
    col+=float3(1.,.9,.75)*2.2*(1.-smoothstep(-r,r,sdRoundRect(mq,float2(.15,.17),.03)));
    if(u[34]<.5)return col;
    // Seen from a seat, the far side of the room sits just below eye level:
    // a counter with a dark vinyl front between chrome kick plate and top edge.
    float back=smoothstep(.3f,-.2f,d.z/max(cs,.01f));
    float lineW=r*.6+.004;
    col+=back*(float3(.020,.007,.006)*smoothstep(-.215f,-.2f,el)*smoothstep(-.05f,-.065f,el)
              +float3(.13,.105,.085)*(exp(-pow((el+.06)/lineW,2.))+.5*exp(-pow((el+.205)/lineW,2.))));
    // Stools before it: red seats with a chrome rim catching the lights.
    float sx=wrapAngle(az+.05)/.17,so=(fract(sx)-.5)*.17*cs;
    float seat=1.-smoothstep(-r,r,sdRoundRect(float2(so,el+.128),float2(.048,.013),.012));
    float rim=exp(-pow((el+.115)/(r*.5+.003),2.))*(1.-smoothstep(.035f,.05f+r,abs(so)));
    col+=back*(float3(.05,.008,.006)*seat+float3(.20,.16,.12)*rim);
    // Pie case on the counter: warm interior, bright shelf edges.
    float2 pq=float2(wrapAngle(az+2.2)*cs,el+.017);
    float box=1.-smoothstep(-r,r,sdRoundRect(pq,float2(.21,.042),.01));
    float shelves=exp(-pow((pq.y+.022)/(r*.5+.003),2.))+exp(-pow((pq.y-.018)/(r*.5+.003),2.));
    col+=box*float3(.95,.62,.34)*(.10+.12*shelves);
    // Window onto the night street toward -z.
    float2 wq=float2(wrapAngle(az+1.5708)*cs,el-.17);
    float window=1.-smoothstep(-r,r,sdRoundRect(wq,float2(.62,.23),.02));
    col+=window*float3(.008,.010,.015);
    // Red neon sign (matches the red light from -x), teal OPEN tubes in the window.
    float breath=.97+.03*sin(u[2]*.7);
    float2 sq=float2(wrapAngle(az-2.95)*cs,el-.17);
    float sign=min(abs(sdRoundRect(sq,float2(.13,.055),.03)),
                   max(abs(sq.y-.014*sin(sq.x*90.+1.)),abs(sq.x)-.09));
    col+=float3(1.,.08,.04)*breath*(1.6*neonTube(sign,r)+.05*exp(-sign/.06));
    float2 oq=float2(wrapAngle(az+1.25)*cs,el-.1);
    float open=abs(length(oq/float2(1.,.55))-.06);
    col+=float3(.05,.9,.8)*(.8*neonTube(open,r)+.02*exp(-open/.05));
    // Scattered out-of-focus lamps and chrome glints: at most one disc per cell.
    const float N=44.;
    float2 g=float2((az+3.1415927)*N/6.2831853,(el+.25)/.1);
    float2 cell=floor(g);
    if(cell.y>=0. && cell.y<6.) {
        float2 h=hash22(float2(fmod(cell.x,N),cell.y)+3.);
        if(h.x>.55) {
            float size=6.2831853/N,margin=max(0.f,.5-r/size);
            float2 c=(cell+.5+(h-.5)*margin*2.)*float2(size,.1)-float2(3.1415927,.25);
            float2 o=float2(wrapAngle(az-c.x)*cs,el-c.y);
            float k=hash21(cell+.7);
            float3 tone=k<.5?float3(1.,.70,.42):(k<.7?float3(1.,.2,.1):(k<.85?float3(.1,.85,.75):float3(.85,.9,1.)));
            float disc=length(o);
            col+=tone*(.04+.16*h.y)*(1.-smoothstep(r*.88-.0015,r,disc))*(.8+.35*smoothstep(.3*r,r,disc))*(1.-.6*window);
        }
    }
    // Street lamps through the glass, and the headlights of a passing car.
    float2 sg=float2(wq.x/.21,(wq.y+.08)/.16);float2 sc=floor(sg);
    float2 sh=hash22(sc+11.);
    float2 sl=float2((sg.x-sc.x-.5-(sh.x-.5)*.3)*.21,(sg.y-sc.y-.5)*.16);
    col+=window*float3(1.,.58,.24)*.35*step(.6f,sh.y)*(1.-smoothstep(r*.88-.0015,r,length(sl)));
    float fade=headlightFade(u);
    if(fade>0.)for(int k=0;k<2;k++) {
        float3 l=normalize(headlightLamp(u,k?1.4:-1.4)-origin);
        float a=length(cross(d,l))*step(0.f,dot(d,l));
        col+=window*fade*float3(1.,.97,.92)*2.2*(1.-smoothstep(r*.88-.0015,r,a));
    }
    return col;
}
// Reflected surroundings: the defocused diner near the horizon, a warm
// ceiling above, its light box, and the off-camera neon tubes.
float3 environment(float3 d, float ambient, device const float *u) {
    float3 env=backdrop(d,BACKDROP_BLUR,float3(0,.6,0),u);
    env=mix(env,mix(float3(.06,.05,.042),float3(.21,.18,.15),smoothstep(.5f,.95f,d.y)),smoothstep(.3f,.65f,d.y));
    float2 a=float2(d.x/max(.05f,d.y),d.z/max(.05f,d.y));
    float panel=(1.-smoothstep(.37f,.42f,abs(a.x+.58)))*(1.-smoothstep(.5f,.56f,abs(a.y+.4)));
    float bars=1.-.38*exp(-abs(a.x+.58)*120.)-.28*exp(-abs(a.y+.4)*120.);
    env+=float3(1.4,1.18,.83)*panel*bars*mix(.6f,1.f,ambient);
    if(u[34]>.5) {
        // Off-camera neon tubes have finite width, so reflections slide over
        // moving liquid and ceramic rather than being painted onto the image.
        float red=exp(-pow((d.x+.72)*17.,2.))*exp(-pow((d.y-.22)*2.7,4.));
        float teal=exp(-pow((d.z+.76)*22.,2.))*exp(-pow((d.y-.30)*3.,4.));
        float breath=.97+.03*sin(u[2]*.7);
        env+=float3(.9,.055,.03)*red*breath+float3(.008,.11,.10)*teal;
    }
    return env;
}
float3 keyDir() {return normalize(float3(-.65,1.3,-.7));}
// Warm key and cool sky ambient: shade is modelled by temperature, not gray fill.
float3 keyLight(float room) {return mix(float3(1.05,.59,.32),float3(1.15,.97,.76),room)*1.35;}
float3 ambientLight(float3 n,float room) {
    float3 a=mix(float3(.07,.06,.05),float3(.16,.18,.22),smoothstep(-.6f,.9f,n.y));
    a+=float3(.07,.072,.078)*max(0.f,dot(n,normalize(float3(.2,.6,1.))));
    // Warm bounce from the lamp-lit walnut and saucer below.
    a+=float3(.05,.034,.022)*smoothstep(.4f,-.6f,n.y);
    return a*(room*.6+.6);
}
// Soft key-light occlusion; 0 is a full umbra, the ambient term lights it.
float shadow(float3 p,float3 light,CupBody body) {
    float result=1., t=.025;
    for(int i=0;i<36;i++) {
        float d=occluder(p+light*t,body);
        if(d<.0008) return 0.;
        result=min(result,12.*d/t); t+=clamp(d,.014f,.18f);
        if(t>3.)break;
    }
    return smoothstep(0.f,1.f,clamp(result,0.f,1.f));
}
// Table/saucer variant: spilled liquid casts a partial contact shadow too.
float shadowSpill(float3 p,float3 light,CupBody body,device const float *field) {
    float result=1., t=.025, liquid=1.;
    for(int i=0;i<36;i++) {
        float3 q=p+light*t;
        float d=occluder(q,body), f=fluidSample(q,field).x;
        if(d<.0008) return 0.;
        if(f<.002)liquid=.45;
        result=min(result,12.*d/t); t+=clamp(min(d,max(f,.004f)),.006f,.18f);
        if(t>3.)break;
    }
    return smoothstep(0.f,1.f,clamp(result,0.f,1.f))*liquid;
}
// Short taps resolve the creases where the foot meets the saucer and the
// saucer rim meets the table; longer taps give the broad bounce occlusion.
float contactOcclusion(float3 p,float3 n,CupBody body) {
    const float d[5]={.012,.03,.06,.11,.18},w[5]={.3,.25,.2,.15,.1};
    float ao=1.;
    for(int i=0;i<5;i++) {
        float3 q=p+n*d[i];
        ao-=w[i]*max(0.f,d[i]-min(occluder(q,body),q.y+.13))/d[i];
    }
    return clamp(ao,.12f,1.f);
}
// Normalized GGX with height-correlated Smith visibility and Schlick Fresnel,
// premultiplied by pi to match the unnormalized-albedo diffuse convention.
float ggxSpecular(float3 n,float3 v,float3 l,float a,float f0) {
    float3 h=normalize(l+v);
    float nl=max(dot(n,l),0.f),nv=max(dot(n,v),1e-4f),nh=max(dot(n,h),0.f),vh=max(dot(v,h),0.f);
    float a2=a*a,q=nh*nh*(a2-1.)+1.;
    float vis=.5/(nl*sqrt(nv*nv*(1.-a2)+a2)+nv*sqrt(nl*nl*(1.-a2)+a2)+1e-5);
    return a2/(q*q)*vis*(f0+(1.-f0)*pow(1.-vh,5.f))*nl;
}
// Anisotropic GGX (alpha ax along tangent t), same convention as above.
float ggxAniso(float3 n,float3 t,float3 v,float3 l,float ax,float ay,float f0) {
    float3 h=normalize(l+v),b=normalize(cross(n,t));t=cross(b,n);
    float nl=max(dot(n,l),0.f),nv=max(dot(n,v),1e-4f),vh=max(dot(v,h),0.f);
    float3 m=float3(dot(h,t)/ax,dot(h,b)/ay,dot(h,n));
    float q=dot(m,m),a=sqrt(ax*ay);
    float vis=.5/(nl*sqrt(nv*nv*(1.-a*a)+a*a)+nv*sqrt(nl*nl*(1.-a*a)+a*a)+1e-5);
    return 1./(ax*ay*q*q)*vis*(f0+(1.-f0)*pow(1.-vh,5.f))*nl;
}
float fresnelRough(float nv,float f0,float rough) {
    return f0+(max(1.-rough,f0)-f0)*pow(1.-clamp(nv,0.f,1.f),5.f);
}
// Value noise averaged toward its mean once a cell is under ~2 pixels wide.
float filteredNoise(float2 p,float footprint) {
    return mix(noise(p),.5,smoothstep(.5f,1.5f,footprint));
}
// Thin-lens blur (radians) behind the subject only: the near saucer never smears.
float defocus(float t,float focus) {return .42*max(0.f,1./(focus+.9)-1./t);}
// Flat-sawn walnut planks ~4.5 cm wide. Each board is a slice through a
// tapering, wandering log, so growth rings meet its face as nested cathedral
// arches; pores streak along the grain (x). Every term fades to its mean
// once its period drops under ~2 pixels (fw: world footprint).
float3 walnut(float2 p,float fw) {
    const float W=.45,L=3.4,RINGS=30.;
    float row=floor(p.y/W),across=p.y-row*W;
    float along=p.x+hash21(float2(row,3.7))*29.,board=floor(along/L),l=along-board*L;
    float2 id=float2(row,board);
    float h1=hash21(id+.17),h2=hash21(id+5.3),h3=hash21(id+9.1);
    // The pith stays 2-5 cm below the face, so arches never close into bullseyes.
    float lateral=across-W*(.15+.7*h1)+.04*sin(l*.6+h2*6.);
    float depth=.22+.25*h2+(l-L*.5)*(.02+.035*h3)*(h1>.5?1.:-1.)+.02*sin(l*1.1+h3*5.);
    float radius=sqrt(lateral*lateral+depth*depth)
                +.025*noise(float2(l*.9,across*6.)+h2*17.)+.008*noise(float2(l*3.,across*24.));
    float f=fract(radius*RINGS);
    float late=smoothstep(.55f,.9f,f)*(1.-smoothstep(.93f,1.f,f));
    late=mix(late,.3,smoothstep(.35f,1.f,fw*RINGS*(abs(lateral)/radius+.1)));
    float pores=filteredNoise(float2(along*2.5,across*160.),fw*160.);
    float fibre=filteredNoise(float2(along*.9,across*48.),fw*48.);
    float3 col=mix(float3(.075,.041,.026),float3(.026,.014,.010),late*.7);
    col*=(.86+.28*fibre)*(1.06-.12*pores);
    // Boards differ in tone; some lean grey-violet, as walnut does.
    col*=.8+.35*h3;
    col=mix(col,dot(col,float3(.3,.59,.11))*float3(1.06,.97,.97),.25*h2);
    float seamWidth=max(.006f,fw*.7);
    float edge=min(min(across,W-across),min(l,L-l));
    return col*(1.-.5*(1.-smoothstep(seamWidth*.5,seamWidth*1.5,edge))*(.006/seamWidth));
}
// Faint glaze crazing: Voronoi cell borders (F2-F1), 3% darker.
float crazing(float2 q,float period,float fw) {
    float fade=1.-smoothstep(.15f,.5f,fw*12.);
    if(fade<=0.)return 0.;
    float2 i=floor(q),f=fract(q);float f1=9.,f2=9.;
    for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++) {
        float2 c=i+float2(x,y),o=float2(x,y)+hash22(float2(period>0.?fmod(c.x+period*64.,period):c.x,c.y)+19.)-f;
        float d=dot(o,o);
        if(d<f1){f2=f1;f1=d;} else if(d<f2)f2=d;
    }
    return fade*(1.-smoothstep(0.f,.04f+fw*12.,sqrt(f2)-sqrt(f1)));
}
// fw: world-space pixel footprint at the hit, used to band-limit patterns.
float3 material(float3 p, int mat, float fw, device const float *u, CupBody body) {
    if(mat==0)p=cupLocal(p,body);
    float3 cup=float3(u[12],u[13],u[14]), saucer=float3(u[21],u[22],u[23]);
    if(mat==0 || mat==1) {
        float3 col=mat==0?cup:saucer;
        if(u[27]>.5) {
            // Stripe edges widen with the footprint, keeping their area.
            float edge=fw*.7;
            float stripe=mat==0 ? (1.-smoothstep(.010f-edge,.014f+edge,abs(p.y-.862)))*(1.-smoothstep(.84f,.87f,length(p.xz)))
                               : (1.-smoothstep(.010f-edge,.017f+edge,abs(length(p.xz)-1.14)));
            col=mix(col,float3(.27,.045,.055),stripe);
            col*=.975+.025*filteredNoise(p.xz*83.+p.y*13.,fw*83.);
        }
        float a=atan2(p.z,p.x);
        col*=1.-.03*(mat==0 ? crazing(float2(a*40./6.2831853,p.y*12.),40.,fw) : crazing(p.xz*12.,0.,fw));
        if(mat==0) {
            // Unglazed foot ring: matte biscuit where the cup stands.
            col*=mix(float3(1),float3(.86,.79,.70),smoothstep(-.016f,-.03f,p.y)*(1.-smoothstep(.56f,.6f,length(p.xz))));
            // One old dried drip down the outside, darker at its pinned edges.
            float outer=cupInnerRadius(p.y)+.096;
            float x=wrapAngle(a-1.62)*.8-.012*sin(p.y*11.+1.)-.006*sin(p.y*29.);
            float width=.021*(.55+.45*smoothstep(.55f,.95f,p.y));
            float drop=length(float2(x,(p.y-.55)*.8))-.021;
            float d=min(max(abs(x)-width,max(.55-p.y,p.y-.99)),drop);
            float stain=(1.-smoothstep(-fw,fw,d))*step(outer-.03,length(p.xz))*step(p.y,1.01);
            float ring=1.-smoothstep(.002f+fw,.006f+fw,-d);
            col*=mix(float3(1),mix(float3(.93,.85,.72),float3(.80,.66,.50),ring),stain);
        }
        return col;
    }
    if(mat==3) return walnut(p.xz,fw);
    if(mat==5) return float3(.62,.60,.57); // stainless F0
    if(mat==6) {
        // Sugar packet: paper, printed band, crimped sealed ends.
        float2 q=packetLocal(p);
        float3 col=float3(.66,.64,.60);
        float band=1.-smoothstep(.06f-fw,.06f+fw,abs(q.x+.03));
        col=mix(col,float3(.45,.05,.05),band);
        float crimp=smoothstep(.215f,.225f,abs(q.x))*(.5+.5*cos(q.y*180.)*(1.-smoothstep(.3f,.8f,fw*30.)));
        return col*(1.-.12*crimp);
    }
    float3 coffee=float3(u[15],u[16],u[17])*.48;
    // APIC cream is exclusively advected material, never decorative noise.
    if(u[32]>.5)return coffee/.48;
    float r=length(p.xz), a=atan2(p.z,p.x);
    // Beer-Lambert to the glaze: the shallow meniscus wedge transmits amber.
    float3 density=-log(clamp(float3(u[15],u[16],u[17]),.0001f,.9999f));
    float neutral=min(density.x,min(density.y,density.z));
    float3 absorb=neutral*1.25+(density-neutral)*3.5;
    coffee=max(coffee,cup*.85*exp(-absorb*2.*max(.745-r,0.f)*2.5));
    float swirls=noise(float2(a*3.+u[2]*.13,r*19.+sin(a*3.+u[2]*.22)*.6));
    float cream=smoothstep(.58f,.735f,r)*(.40+.60*swirls);
    cream+=.15*smoothstep(.67f,.9f,swirls)*smoothstep(.3f,.6f,r);
    cream=clamp(cream,0.f,.9f);
    // Partly mixed cream is cafe-au-lait, not chalk: coffee tints it first.
    return mix(coffee,float3(u[18],u[19],u[20])*mix(float3(.62,.48,.36),float3(1),cream*cream),cream);
}
// Glossy surfaces mirror their neighbours: the cup shows the saucer and the
// table, the varnish shows the cup. A short trace against the crockery and
// the tabletop, shaded with key and ambient only (no shadows, no recursion).
bool nearSphere(float3 p,float3 r,float3 c,float radius) {
    float3 o=p-c;float b=dot(o,r);return b*b-dot(o,o)+radius*radius>0. && (b<0. || dot(o,o)<radius*radius);
}
// Rough coats blur what they mirror with distance; that fades it to the environment.
float3 reflected(float3 p,float3 r,float fw,float rough,float room,device const float *u,CupBody body) {
    float plane=r.y<-1e-4 ? -(p.y+.13)/r.y : 1e9;
    float3 light=keyDir();
    if(nearSphere(p,r,float3(0,-.05,0),1.33) || nearSphere(p,r,body.position.xyz,1.25)) {
        float t=.012;
        for(int i=0;i<32;i++) {
            float3 q=p+r*t;float2 h=crockery(q,body);
            if(h.x<.002) {
                float3 n=solidNormal(q,body);
                float3 seen=material(q,int(h.y),fw+t*.004,u,body)*(ambientLight(n,room)+keyLight(room)*max(0.f,dot(n,light))*.75);
                return mix(seen,environment(r,room,u),smoothstep(0.f,1.f,t*rough*10.));
            }
            t+=max(h.x*.9,.006f);if(t>min(plane,3.f))break;
        }
    }
    if(plane<8.) {
        float3 q=p+r*plane;
        if(sdRoundRect(q.xz,float2(TABLE_HALF),.35)<0.)
            return walnut(q.xz,fw+plane*.01)*(ambientLight(float3(0,1,0),room)+keyLight(room)*.6)*mix(1.f,.55f,smoothstep(1.5f,5.f,length(q.xz)));
    }
    return environment(r,room,u);
}
float3 neonFill(float3 n,device const float *u) {
    if(u[34]<.5)return 0.;
    return float3(.13,.009,.005)*max(0.f,dot(n,RED_DIR))+float3(.005,.048,.043)*max(0.f,dot(n,TEAL_DIR));
}
// Passing car: two beams sweep the diorama through the window. Returns the
// unshadowed radiance reaching p and its direction. The window mullions
// (plane z=-5) act as a gobo; the lamp's apparent size there sets the penumbra.
float3 headlight(float3 p,device const float *u,thread float3 &l) {
    l=float3(0,1,0);
    float fade=headlightFade(u);
    if(u[34]<.5 || fade<=0.)return 0.;
    float travel=headlightTravel(u);
    float3 lamp=headlightLamp(u,0.);
    l=normalize(lamp-p);
    float beam=exp(-pow((p.x-travel*.63-.15)*2.4,2.))+.7*exp(-pow((p.x-travel*.63+.65)*2.4,2.));
    float s=clamp((-5.-p.z)/(lamp.z-p.z),0.f,1.f),blur=.01+.12*s;
    float2 g=(p+(lamp-p)*s).xy;
    float bar=abs(fract(g.x/.7+.5)-.5)*.7;
    float gobo=smoothstep(.05-blur,.05+blur,bar)*smoothstep(.04-blur,.04+blur,abs(g.y-1.9))*smoothstep(.35-blur,.35+blur,g.y);
    return float3(1.,.975,.93)*fade*beam*gobo;
}
float3 coffeeAbsorption(device const float *u) {
    // Theme color controls absorption, not an opaque diffuse paint layer.
    float3 opticalDensity=-log(clamp(float3(u[15],u[16],u[17]),.0001f,.9999f));
    float neutral=min(opticalDensity.x,min(opticalDensity.y,opticalDensity.z));
    return neutral*1.25+(opticalDensity-neutral)*3.5;
}
// Semi-infinite multiple-scattering reflectance for single-scatter albedo a:
// coffee absorbed between many scattering events turns part-mixed cream tan.
float3 multipleScatter(float3 a) {float3 s=sqrt(max(1.-a,0.f));return (1.-s)/(1.+s);}
// One bounded refracted path through the actual reconstructed liquid. Coffee
// absorbs light; only added cream scatters it. No opaque brown surface coat.
float3 transmittedSolid(float3 p,int mat,float3 rd,float fw,device const float *u,CupBody body,
                        device const uint4 *film,device const uint4 *dry,device const float2 *waves) {
    float3 n=solidNormal(p,body),l=keyDir();
    float3 base=material(p,mat,fw,u,body);
    float4 coating=0.;
    if((mat==1||mat==3)&&abs(p.y-filmBaseHeight(p.xz))<.025) {
        coating=sampleFilm(p.xz,film,dry);base=stainColor(base,coating);
        if(coating.x>.0001)n=wetNormal(p,n,film,waves);
    }
    float shade=shadow(p+n*.008,l,body);
    float3 hl,head=headlight(p,u,hl);
    float3 col=base*(ambientLight(n,u[10])+keyLight(u[10])*max(0.f,dot(n,l))*shade
                     +neonFill(n,u)+head*max(0.f,dot(n,hl)));
    float wet=smoothstep(.00008f,.0015f,coating.x);
    return mix(col,environment(reflect(rd,n),u[10],u),wet*fresnelRough(dot(-rd,n),.02,.1));
}
float3 coffeeTransmission(float3 p,float3 n,float3 rd,float fw,float keyShade,device const float *u,
                         device const float *field,CupBody body,
                         device const uint4 *film,device const uint4 *dry,device const float2 *waves) {
    float3 ray=refract(rd,n,1./1.333),q=p-n*.004;
    float3 transmission=1.,glow=1.,radiance=0.;bool entered=false,exited=false,bounced=false;
    float3 coffeeAbsorb=coffeeAbsorption(u);
    float3 light=keyDir();
    // Cream in-scattering: key through the surface plus the cool sky dome.
    float3 illumination=ambientLight(float3(0,1,0),u[10])*1.5+keyLight(u[10])*.45*max(0.f,dot(n,light))*keyShade;
    for(int j=0;j<112;j++) {
        float2 ceramic=crockery(q,body);
        if(q.y+.13<ceramic.x)ceramic=float2(q.y+.13,3);
        if(ceramic.x<.0015)
            return radiance+transmission*transmittedSolid(q,int(ceramic.y),ray,fw,u,body,film,dry,waves);
        float4 sample=fluidSample(q,field);
        if(!exited && sample.x<.0015) {
            entered=true;
            float step=min(.018f,max(.003f,ceramic.x*.8));
            float cream=clamp(sample.y,0.f,1.f);
            float3 absorb=coffeeAbsorb*(1.-cream);
            float3 scatter=float3(32.)*cream;
            float3 extinction=absorb+scatter+1e-5;
            float3 attenuation=exp(-extinction*step);
            // Cream in-scattering is seen through a softened coffee absorption,
            // so a cloud below the surface reads as a tan glow fading with depth.
            radiance+=glow*(1.-exp(-scatter*step))*multipleScatter(scatter/extinction)
                     *float3(.92,.86,.72)*illumination;
            transmission*=attenuation;glow*=exp(-(mix(absorb,float3(dot(absorb,float3(1./3.))),.6)*.3+scatter)*step);
            if(max(glow.x,max(glow.y,glow.z))<.003)return radiance;
            q+=ray*step;
        } else {
            if(entered && !exited) {
                // Exit refraction; a single internal reflection is allowed.
                float e=.008;
                float3 en=float3(fluidSample(q+float3(e,0,0),field).x-fluidSample(q-float3(e,0,0),field).x,
                                 fluidSample(q+float3(0,e,0),field).x-fluidSample(q-float3(0,e,0),field).x,
                                 fluidSample(q+float3(0,0,e),field).x-fluidSample(q-float3(0,0,e),field).x);
                en=length(en)>1e-6?normalize(en):ray;
                float3 outRay=refract(ray,-en,1.333);
                if(length(outRay)<1e-6) {
                    if(bounced)return radiance; // bounded multiple scattering
                    bounced=true;q-=ray*.020;ray=reflect(ray,en);continue;
                }
                ray=outRay;exited=true;
            }
            q+=ray*clamp(min(ceramic.x*.75,max(.006f,sample.x*.65)),.003f,.18f);
        }
        if(length(q-p)>5.)break;
    }
    return radiance+transmission*environment(ray,u[10],u);
}
// The free surface comes from the particle field alone: fluidDistance also
// folds in the ceramic, so its differences straddle the wall into speckles.
// Hits clipped by the ceramic sit just off its face, so fall back to it.
float3 liquidNormal(float3 p,device const float *field,CupBody body) {
    float3 g=fluidGradient(p,field);
    return length(g)>1e-3 ? normalize(g) : solidNormal(p,body);
}
// Micro-bubbles trapped in the meniscus ring (outer ~3% of the radius): at
// most one tiny sphere per Voronoi cell, denser against the glaze. Once a
// bubble is under ~a pixel they average into a paler band instead.
float meniscusBubbles(float3 p,thread float3 &n,float fw,CupBody body) {
    float3 q=cupLocal(p,body);
    float inner=cupInnerRadius(q.y),gap=inner-length(q.xz);
    float4 back=float4(-body.rotation.xyz,body.rotation.w);
    if(gap>.035 || gap<-.012 || rotateQ(back,n).y<.6)return 0.;
    const float cell=.016;
    float ring=floor(6.2831853*inner/cell);
    float2 g=float2((atan2(q.z,q.x)/6.2831853+.5)*ring,gap/cell);
    float2 i=floor(g);
    float best=9.;float2 offset=0.;
    for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++) {
        float2 c=i+float2(x,y);
        float2 h=hash22(float2(fmod(c.x+ring,ring),c.y+40.));
        if(h.x>.9*exp(-max(c.y,0.f)*cell/.014))continue;
        float radius=.2+.22*h.y;
        float2 o=(g-(c+.5+(h-.5)*.35))/radius;
        float d=length(o);
        if(d<1. && d<best){best=d;offset=o;}
    }
    float detail=1.-smoothstep(.5f,1.2f,fw/(cell*.6));
    float band=exp(-max(gap,0.f)/.012);
    if(best<1. && detail>0.) {
        float3 tangent=normalize(float3(-q.z,0,q.x)),radial=normalize(float3(q.x,0,q.z));
        float3 bn=offset.x*tangent-offset.y*radial+sqrt(max(0.f,1.-best*best))*float3(0,1,0);
        n=normalize(mix(n,rotateQ(body.rotation,bn),detail));
        return .3*detail;
    }
    return band*.12*(1.-detail);
}
float3 shadeHit(float3 p,float3 rd,int mat,float fp, device const float *u,device const float *w, CupBody body,
                device const uint4 *film,device const uint4 *dry,device const float2 *waves,
                device const float4 *impacts) {
    bool real=u[32]>.5;
    float3 n=mat==2&&real ? liquidNormal(p,w,body) : normalAt(p,u,w,body), light=keyDir(), v=-rd;
    float4 coating=0.;float wet=0.;
    if(real && (mat==1||mat==3) && abs(p.y-filmBaseHeight(p.xz))<.025) {
        coating=sampleFilm(p.xz,film,dry);wet=smoothstep(.00008f,.0015f,coating.x);
        if(wet>0.)n=normalize(mix(n,wetNormal(p,n,film,waves),wet));
    }
    if(mat==2 && real && inCup(p,body)) {
        float3 local=cupLocal(p,body);float2 gradient=0.;
        for(int j=0;j<64;j++) {
            float4 impact=impacts[j];float age=u[35]-impact.z;
            if(impact.w<=0. || age<0. || age>1.5)continue;
            float2 delta=local.xz-impact.xy;float r=length(delta),front=age*1.25,q=(r-front)/.05;
            float phase=(r-front)*80.;
            float derivative=impact.w*exp(-age*3.-q*q)*(80.*cos(phase)-2.*q/.05*sin(phase));
            gradient+=delta/max(r,.005f)*derivative;
        }
        n=normalize(n-rotateQ(body.rotation,float3(gradient.x,0,gradient.y)));
    }
    float nv=max(dot(n,v),1e-4f);
    float fw=fp/max(.35f,nv);
    float3 base=stainColor(material(p,mat,fw,u,body),coating);
    float milk=0.;
    if(mat==2 && real)milk=clamp(fluidSample(p-n*.008,w).y,0.f,1.f);
    // One key-light shadow ray shared by diffuse, specular and in-scattering.
    float key=real&&(mat==1||mat==3) ? shadowSpill(p+n*.006,light,body,w) : shadow(p+n*.006,light,body);
    float ao=contactOcclusion(p,n,body);
    float room=u[10];
    float3 warm=keyLight(room),hl,head=headlight(p,u,hl);
    if(dot(head,head)>0.)head*=shadow(p+n*.009,hl,body);
    float ndl=dot(n,light);
    float3 irradiance=ambientLight(n,room)*ao+warm*max(0.f,ndl)*key+neonFill(n,u)*mix(.4f,1.f,ao)+head*max(0.f,dot(n,hl));
    float3 col=base*irradiance;
    if(mat==5) {
        // Polished steel: no diffuse, coloured Schlick Fresnel.
        float3 F=base+(1.-base)*pow(1.-nv,5.f);
        col=reflected(p+n*.004,reflect(rd,n),fw,.14,room,u,body)*F*mix(.7f,1.f,ao)
           +base*(warm*ggxSpecular(n,v,light,.14,1.)*key+head*ggxSpecular(n,v,hl,.14,1.));
        return col;
    }
    if(mat<=1) {
        // Glaze over ceramic body. The body scatters: a soft wrap past the
        // terminator, and warm light leaking out where the key exits the
        // ceramic within a few millimetres (the rolled rim, the handle).
        float4 back=float4(-body.rotation.xyz,body.rotation.w);
        float3 local=mat==0?cupLocal(p,body):p,nl=mat==0?rotateQ(back,n):n,ll=mat==0?rotateQ(back,light):light;
        float3 pin=local-nl*.012;
        float s1=mat==0?cupShape(pin+ll*.04):saucerShape(pin+ll*.04),s2=mat==0?cupShape(pin+ll*.09):saucerShape(pin+ll*.09);
        // Only the rim and handle are thin enough; deeper, the "exit" is the dark interior.
        float thin=mat==0?max(smoothstep(.9f,.98f,local.y),smoothstep(.85f,.9f,length(local.xz))):smoothstep(1.12f,1.24f,length(p.xz));
        float through=(smoothstep(-.03f,.008f,s1)*.6+smoothstep(-.03f,.008f,s2)*.4)*thin;
        float bisque=mat==0?smoothstep(-.016f,-.03f,local.y)*(1.-smoothstep(.56f,.6f,length(local.xz))):0.;
        float wrap=.13*smoothstep(-.35f,.2f,ndl)*mix(.35f,1.f,key);
        col+=base*warm*float3(1.,.62,.36)*(wrap+.4*through*smoothstep(.25f,-.2f,ndl))*ao*(1.-bisque);
        // Clearcoat: F0 .04, sharp, with a faint orange-peel ripple of its own.
        float3 nc=n;
        float peel=.012*(1.-smoothstep(.5f,1.5f,fw*22.));
        if(peel>0.) {
            float3 q=local*22.;float c=noise3(q);
            float3 g=float3(noise3(q+float3(.35,0,0)),noise3(q+float3(0,.35,0)),noise3(q+float3(0,0,.35)))-c;
            if(mat==0)g=rotateQ(body.rotation,g);
            nc=normalize(n-(g-n*dot(g,n))*peel/.35);
        }
        float coat=1.-bisque,rough=mix(.09f,.05f,wet);
        float F=fresnelRough(dot(nc,v),.04,rough)*coat;
        col=col*(1.-F)+reflected(p+n*.004,reflect(rd,nc),fw,rough,room,u,body)*F;
        col+=coat*(warm*ggxSpecular(nc,v,light,rough,.04)*key+head*ggxSpecular(nc,v,hl,rough,.04));
        col+=bisque*warm*ggxSpecular(n,v,light,.55,.04)*key;
        return col;
    }
    if(mat==2) {
        if(real)col=coffeeTransmission(p,n,rd,fw,key,u,w,body,film,dry,waves);
        float foam=meniscusBubbles(p,n,fw,body);
        col=mix(col,float3(.62,.52,.40)*irradiance,foam);
        nv=max(dot(n,v),1e-4f);
        float3 refl=reflect(rd,n);
        float3 env=environment(refl,room,u);
        // Reflect the inner ceramic wall, not just a painted highlight.
        float t=.025;
        for(int i=0;i<24;i++) {
            float3 rp=p+n*.009+refl*t; float2 h=crockery(rp,body);
            if(h.x<.004) {env=material(rp,int(h.y),fw,u,body)*(.3+.35*max(0.f,rp.y));break;}
            t+=max(.008f,h.x*.85); if(t>2.)break;
        }
        // Faint oil film: thin-film interference tints grazing reflections.
        float3 local=cupLocal(p,body);
        float thick=1.3+.6*noise(local.xz*7.);
        float3 sheen=.5+.5*cos(6.2831853*(thick*(1.6-nv)+float3(0,.33,.67)));
        env*=mix(float3(1),.6+.8*sheen,.25*pow(1.-nv,2.f));
        float rough=mix(.08f,.16f,milk);
        col=mix(col,env,fresnelRough(nv,.02,0.));
        col+=warm*ggxSpecular(n,v,light,rough,.02)*key+head*ggxSpecular(n,v,hl,rough,.02);
        return col;
    }
    if(mat==6) return col+warm*ggxSpecular(n,v,light,.45,.04)*key;
    // Walnut under a varnish clearcoat. Fibres along x scatter the coat's
    // sheen across the grain (rougher across, sharp along), so headlight and
    // key highlights streak over the planks and glide as the lamp moves.
    // The key is a compact lamp almost mirrored toward the camera: a broad
    // GGX tail would grey the whole near table, so its glint stays tight.
    float rough=mix(.06f,.04f,wet),glint=.014;
    float F=fresnelRough(nv,mix(.04f,.02f,wet),rough);
    col=col*(1.-F)+reflected(p+n*.004,reflect(rd,n),fw,rough*1.5,room,u,body)*F*mix(.6f,1.f,ao);
    float3 grain=float3(1,0,0);
    col+=warm*key*(ggxSpecular(n,v,light,glint,.04)+.15*ggxAniso(n,grain,v,light,.08,.3,.04))
        +head*(ggxSpecular(n,v,hl,rough,.04)+.3*ggxAniso(n,grain,v,hl,.08,.3,.04));
    // The key is a lamp over the table: its pool falls off toward the edges.
    float r=length(p.xz);
    col*=mix(1.f,.55f,smoothstep(1.5f,5.f,r));
    // The tabletop ends; beyond its defocused edge lies the diner.
    float edge=sdRoundRect(p.xz,float2(TABLE_HALF),.35),soft=max(fp*1.5,fw);
    col+=warm*.02*exp(-pow((edge+.05)/(soft+.03),2.));
    return mix(col,backdrop(rd,BACKDROP_BLUR,p,u),smoothstep(-soft,soft,edge));
}
float3 shadeSpray(float3 p,float3 rd,SprayHit hit,float fp,device const float *u,device const float *field,
                  CupBody body,device const uint4 *film,device const uint4 *dry,
                  device const float2 *waves,device const float4 *impacts) {
    float3 n=hit.normal,inside=refract(rd,n,1./1.333);
    float thickness=max(.001f,2.*hit.radius*dot(-inside,n));
    float3 exitPoint=p+inside*thickness,exitNormal=normalize(exitPoint-(p-n*hit.radius));
    float3 ray=refract(inside,-exitNormal,1.333);
    if(length(ray)<1e-6)ray=reflect(inside,exitNormal);
    float3 background=environment(ray,u[10],u);float t=.004;
    for(int j=0;j<70;j++) {
        float3 q=exitPoint+ray*t;float2 h=scene(q,u,field,body);
        if(h.x<.002) {background=shadeHit(q,ray,int(h.y),fp,u,field,body,film,dry,waves,impacts);break;}
        t+=max(.003f,h.x*.75);if(t>5.)break;
    }
    float3 absorb=coffeeAbsorption(u)*(1.-hit.cream),scatter=32.*hit.cream;
    float3 extinction=absorb+scatter+1e-5,transmission=exp(-extinction*thickness);
    float3 col=background*transmission+(1.-transmission)*multipleScatter(scatter/extinction)*float3(.60,.55,.45);
    col=mix(col,environment(reflect(rd,n),u[10],u),fresnelRough(dot(-rd,n),.02,0.));
    col+=keyLight(u[10])*ggxSpecular(n,-rd,keyDir(),.07,.02);
    return col;
}
// Rising plume: domain-warped value noise scrolling upward, eroded more with
// height so the column breaks into wisps, widening and thinning as it rises.
float steamDensity(float3 p,device const float *u,CupBody body,bool detail) {
    float height=p.y-(u[32]>.5?u[33]:.30+.62*u[3]);
    if(height<0. || height>1.85)return 0.;
    float2 center=-slopes(u)*height+float2(u[11]*.014*height,0)+(u[32]>.5?body.position.xz:float2(0));
    float2 q=p.xz-center;
    float radius=.13+.24*height;
    float envelope=exp(-dot(q,q)/(radius*radius))*smoothstep(0.f,.12f,height)*(1.-smoothstep(.55f,1.85f,height));
    if(envelope<.015)return 0.;
    // 512 lattice units of rise wrap seamlessly (hash31 repeats y every 256).
    float rise=fmod(u[2]*(.9+u[9]*.5),512.f);
    float3 s=float3(q.x*4.5,height*3.-rise,q.y*4.5);
    float2 warp=float2(noise3(s*float3(.55,.5,.55)+float3(0,0,7.3)),noise3(s*float3(.55,.5,.55)+float3(5.1,0,0)))-.5;
    s.xz+=warp*(1.4+height*1.1);
    float n=noise3(s);
    if(detail)n=n*.65+noise3(s*2.+float3(1.7,0,3.1))*.35;
    float wisps=smoothstep(.32f+.2f*height,.66f,n);
    return wisps*envelope*(.85+u[9]*.5)*u[3]*(1.-.3*height);
}
// Henyey-Greenstein (g=.6) blended with isotropic, normalised so isotropic is 1.
float steamPhase(float mu) {return .4+.6*.64/pow(1.36-1.2*mu,1.5f);}
float interleavedGradientNoise(float2 p) {return fract(52.9829189*fract(dot(p,float2(.06711056,.00583715))));}
float3 traceScene(float2 pixel,float stillFrame,device const float *u,device const float *w,CupBody body,
                  device const FluidParticle *fluidParticles,device const int *heads,device const int *next,
                  device const uint *active,device const uint4 *film,device const uint4 *dry,
                  device const float2 *waves,device const float4 *impacts,thread float2 &info) {
    float width=u[0],height=u[1];
    float2 uv=pixel/float2(width,height);
    float yaw=u[4],pitch=u[5]*.8,roll=u[6];
    float3 target=float3(0,.55,0);
    if(u[32]>.5)target.xz=clamp(body.position.xz*.5,float2(-1.6),float2(1.6));
    float upY=1.-2.*(body.rotation.x*body.rotation.x+body.rotation.z*body.rotation.z);
    // ~35 degree vertical field: a longer lens, dollied back to keep framing.
    const float focal=3.;
    float distance=(3.4+(u[32]>.5?.8*(1.-abs(upY))+.25*min(1.6f,length(body.position.xz)):0.))*focal/1.9;
    float3 ro=target+distance*float3(cos(pitch)*sin(yaw),sin(pitch),cos(pitch)*cos(yaw));
    float3 forward=normalize(target-ro),right=normalize(cross(forward,float3(0,1,0))),up=cross(right,forward);
    float3 rr=right*cos(roll)+up*sin(roll),ru=up*cos(roll)-right*sin(roll);
    float2 screen=float2((uv.x*2.-1.)*width/height*.95,(1.-uv.y*2.)*.95-.28);
    float3 rd=normalize(forward*focal+rr*screen.x+ru*screen.y);
    float t=0.,mat=-1.;
    for(int i=0;i<200;i++) {
        float2 hit=scene(ro+rd*t,u,w,body);
        if(hit.x<.0015+t*.0003){mat=hit.y;break;}
        t+=max(.0008f,hit.x*.78);if(t>20.)break;
    }
    // One pixel subtends 1.9/height screen units at the focal length; the
    // defocus blur widens the texture footprint so far grain melts smoothly.
    float pixelAngle=1.9/(height*focal);
    float3 col=backdrop(rd,BACKDROP_BLUR,ro,u);
    SprayHit spray;spray.radius=0.;
    if(u[32]>.5)spray=traceSpray(ro,rd,t,fluidParticles,heads,next,active);
    if(spray.radius>0.) {t=spray.distance;mat=4.;col=shadeSpray(ro+rd*t,rd,spray,t*length(float2(pixelAngle,.6*defocus(t,distance))),u,w,body,film,dry,waves,impacts);}
    else if(mat>=0.)col=shadeHit(ro+rd*t,rd,int(mat),t*length(float2(pixelAngle,.6*defocus(t,distance))),u,w,body,film,dry,waves,impacts);
    else t=20.;
    info=float2(t,mat);
    // Steam: front-to-back Beer-Lambert from a per-pixel jittered start (the
    // temporal pass averages it) plus one light-ward sample for self-shadowing.
    float3 center=float3(body.position.x,1.65,body.position.z);float b=dot(ro-center,rd);
    float disc=b*b-dot(ro-center,ro-center)+1.45*1.45;
    if(disc>0.) {
        float start=max(0.f,-b-sqrt(disc)),end=min(t,-b+sqrt(disc));
        float step=.045;
        float3 light=keyDir(),transmittance=1.,scattered=0.;
        float3 tint=float3(u[24],u[25],u[26]);
        // Droplets scatter forward: steam glows when backlit by the key,
        // the neon, or a passing car's headlights.
        float3 ambient=ambientLight(float3(0,1,0),u[10])*1.6;
        float3 direct=keyLight(u[10])*.5*steamPhase(dot(rd,light));
        if(u[34]>.5) {
            direct+=float3(.9,.06,.03)*.4*steamPhase(dot(rd,RED_DIR))+float3(.02,.30,.27)*.45*steamPhase(dot(rd,TEAL_DIR));
            float3 hl,head=headlight(center-float3(0,1.,0),u,hl);
            direct+=head*1.6*steamPhase(dot(rd,hl));
        }
        start+=step*interleavedGradientNoise(floor(pixel)+5.588238*fmod(stillFrame,64.f));
        for(float d=start;d<end;d+=step) {
            float3 p=ro+rd*d;
            float sigma=steamDensity(p,u,body,true)*4.;
            if(sigma<1e-4)continue;
            float self=exp(-steamDensity(p+light*.12,u,body,false)*4.*.12*3.);
            float3 lit=tint*(ambient+direct*self);
            float alpha=1.-exp(-sigma*step);
            scattered+=transmittance*alpha*lit;transmittance*=1.-alpha;
        }
        col=col*transmittance+scattered;
    }
    return col;
}
// Render state written by coffeePrep and read by every pass of one frame:
// [0..5] camera anchor, [8..14] cup anchor, [26] still frames since reset,
// [27] reset flag, [28..29] sub-pixel jitter, [31] initialised.
#define CAF_RENDER_ARGS device const float *u [[buffer(1)]],device const float *w [[buffer(2)]], \
    device const CupBody &body [[buffer(4)]],device const FluidParticle *fluidParticles [[buffer(5)]], \
    device const int *heads [[buffer(6)]],device const int *next [[buffer(7)]], \
    device const uint *active [[buffer(8)]],device const uint4 *film [[buffer(9)]], \
    device const uint4 *dry [[buffer(10)]],device const float2 *waves [[buffer(11)]], \
    device const float4 *impacts [[buffer(12)]],device float4 *color [[buffer(13)]], \
    device float2 *info [[buffer(14)]],device const float4 *history [[buffer(15)]], \
    device const float *state [[buffer(17)]],uint2 gid [[thread_position_in_grid]]
#define CAF_TRACE(pixel,result) traceScene(pixel,state[26],u,w,body,fluidParticles,heads,next,active,film,dry,waves,impacts,result)
float halton(uint i,uint b) {float f=1.,r=0.;while(i>0){f/=b;r+=f*(i%b);i/=b;}return r;}
// Camera or cup motion beyond ~1/50 pixel since the last reset restarts history;
// anchoring to the reset pose (not the last frame) also catches slow drift.
kernel void coffeePrep(device const float *u [[buffer(1)]],device const CupBody &body [[buffer(4)]],
                       device float *state [[buffer(17)]],uint gid [[thread_position_in_grid]]) {
    if(gid>0)return;
    float camera[6]={u[0],u[1],u[4],u[5],u[6],u[32]};
    float cup[7]={body.position.x,body.position.y,body.position.z,
                  body.rotation.x,body.rotation.y,body.rotation.z,body.rotation.w};
    bool reset=state[31]!=1.;
    for(int k=0;k<6;k++)reset=reset||abs(camera[k]-state[k])>1e-4;
    for(int k=0;k<7;k++)reset=reset||abs(cup[k]-state[8+k])>2e-4;
    if(reset) {
        for(int k=0;k<6;k++)state[k]=camera[k];
        for(int k=0;k<7;k++)state[8+k]=cup[k];
        state[26]=0.;
    } else state[26]=min(state[26]+1.,1e6f);
    uint index=uint(state[26])%16+1;
    state[27]=reset?1.:0.;state[28]=halton(index,2)-.5;state[29]=halton(index,3)-.5;state[31]=1.;
}
kernel void coffee(CAF_RENDER_ARGS) {
    uint width=uint(u[0]),height=uint(u[1]); if(gid.x>=width||gid.y>=height)return;
    float2 jitter=state[27]>.5?float2(0):float2(state[28],state[29]);
    float2 hit;float3 c=CAF_TRACE(float2(gid)+.5+jitter,hit);
    uint i=gid.y*width+gid.x;color[i]=float4(c,1);info[i]=hit;
}
// Without usable history (motion, resets, first frames) supersample only the
// pixels whose 4-neighbours change material or depth: the geometric edges.
kernel void coffeeEdges(CAF_RENDER_ARGS) {
    uint width=uint(u[0]),height=uint(u[1]); if(gid.x>=width||gid.y>=height)return;
    uint i=gid.y*width+gid.x;
    if(state[27]<.5 && history[i].w>=4.)return;
    float2 me=info[i];bool edge=false;
    int2 offsets[4]={int2(1,0),int2(-1,0),int2(0,1),int2(0,-1)};
    for(int k=0;k<4;k++) {
        int2 c=clamp(int2(gid)+offsets[k],int2(0),int2(width-1,height-1));
        float2 o=info[c.y*width+c.x];
        edge=edge||o.y!=me.y||abs(o.x-me.x)>.03*min(o.x,me.x)+.01;
    }
    if(!edge)return;
    // Rotated-grid taps; the existing centre sample is the fifth.
    float2 taps[4]={float2(.125,.375),float2(.375,-.125),float2(-.125,-.375),float2(-.375,.125)};
    float3 sum=color[i].rgb;float2 unused;
    for(int k=0;k<4;k++)sum+=CAF_TRACE(float2(gid)+.5+taps[k],unused);
    color[i]=float4(sum/5.,1);
}
// Khronos PBR Neutral shoulder: hue-preserving, identity below .76. Its toe
// offset is omitted; it would crush this deliberately low-key scene.
float3 neutralTonemap(float3 c) {
    float peak=max(c.r,max(c.g,c.b));
    const float start=.76,desaturation=.15;
    if(peak<start)return c;
    float d=1.-start,newPeak=1.-d*d/(peak+d-start);
    c*=newPeak/peak;
    return mix(c,float3(newPeak),1.-1./(desaturation*(peak-newPeak)+1.));
}
float3 srgbEncode(float3 c) {
    c=clamp(c,0.f,1.f);
    return select(1.055*pow(c,float3(1./2.4))-.055,c*12.92,c<=.0031308);
}
float luminance(float3 c) {return dot(c,float3(.2126,.7152,.0722));}
kernel void coffeeResolve(device uchar *out [[buffer(0)]],device const float *u [[buffer(1)]],
                          device const float4 *dots [[buffer(3)]],device const float4 *color [[buffer(13)]],
                          device const float2 *info [[buffer(14)]],device float4 *history [[buffer(15)]],
                          device float2 *previousInfo [[buffer(16)]],device const float *state [[buffer(17)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint width=uint(u[0]),height=uint(u[1]); if(gid.x>=width||gid.y>=height)return;
    uint i=gid.y*width+gid.x;
    float3 current=color[i].rgb;float4 past=history[i];float2 before=previousInfo[i];
    // With a static camera, history is rejected only when no neighbour still
    // shows the same surface (moving liquid, spray): jittered edges accumulate.
    float3 m1=0.,m2=0.,low=1e9,high=-1e9;bool seen=false;
    for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++) {
        int2 c=clamp(int2(gid)+int2(x,y),int2(0),int2(width-1,height-1));
        uint j=c.y*width+c.x;float3 s=color[j].rgb;float2 o=info[j];
        m1+=s;m2+=s*s;low=min(low,s);high=max(high,s);
        seen=seen||(o.y==before.y&&abs(o.x-before.x)<.04*o.x+.01);
    }
    // Rigid surfaces only change with camera or cup motion, which already
    // resets history; their sub-pixel silhouettes must not reject it.
    seen=seen||(before.y!=2.&&before.y!=4.&&info[i].y!=2.&&info[i].y!=4.);
    float count=state[27]>.5||!seen?0.:min(past.w,255.f);
    float3 result=current;
    if(count>0.) {
        float3 mean=m1/9.,sigma=sqrt(max(m2/9.-mean*mean,0.f));
        float3 clipped=clamp(past.rgb,max(low,mean-1.25*sigma),min(high,mean+1.25*sigma));
        float a=1./(count+1.);
        // Luminance-weighted blend keeps lone specular samples from flickering.
        float wc=a/(1.+luminance(current)),wh=(1.-a)/(1.+luminance(clipped));
        result=(current*wc+clipped*wh)/(wc+wh);
    }
    history[i]=float4(result,count+1.);previousInfo[i]=info[i];
    float3 col=result*u[36];
    float2 uv=(float2(gid)+.5)/float2(width,height);
    // Weather is decorative and local, using the already-fetched weather state.
    if(u[31]==1. || u[31]==2.) {
        float2 grid=uv*float2(80,35);
        grid.y+=u[2]*(u[31]==1.?14.:2.);
        grid.x+=u[31]==1.?grid.y*.13:sin(grid.y*.3+u[2])*.3;
        float2 cell=floor(grid),f=fract(grid)-.5;
        float seed=hash21(cell);
        float mark=u[31]==1.?exp(-f.x*f.x*900.-f.y*f.y*8.):exp(-dot(f,f)*80.);
        float weather=seed>.98?mark*.23*(1.-smoothstep(.35f,.8f,uv.y)):0.;
        col=mix(col,float3(.6,.68,.8),weather);
    }
    // Preserve the existing steam-writing particle choreography.
    for(int k=0;k<int(u[29]);k++) {
        float4 p=dots[k];float2 delta=(uv-p.xy)*float2(width,height);
        float a=exp(-dot(delta,delta)/(p.z*p.z))*p.w;
        col=mix(col,float3(u[24],u[25],u[26]),min(.6f,a));
    }
    col*=1.-(u[34]>.5?.32:.20)*dot(uv-.5,uv-.5);
    col=srgbEncode(neutralTonemap(max(col,0.f)));
    if(u[30]>.5) {
        // Fine static ordered dither: retro texture without temporal noise/flicker.
        constexpr float bayer[16]={0,8,2,10,12,4,14,6,3,11,1,9,15,7,13,5};
        float d=(bayer[(gid.y%4)*4+gid.x%4]+.5)/16.;
        col=floor(col*63.+d)/63.;
    } else {
        // Static triangular +-1 LSB noise: smooth gradients without banding or flicker.
        float2 q=float2(gid);
        col+=(hash21(q)+hash21(q+float2(17.13,59.71))-1.)/255.;
    }
    uint index=(gid.y*width+gid.x)*3;
    out[index]=uchar(clamp(col.r*255.+.5,0.f,255.f));
    out[index+1]=uchar(clamp(col.g*255.+.5,0.f,255.f));
    out[index+2]=uchar(clamp(col.b*255.+.5,0.f,255.f));
}
