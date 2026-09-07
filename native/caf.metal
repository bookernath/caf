#include <metal_stdlib>
using namespace metal;
float fluidDistance(float3 p, device const float *field, CupBody body);
float4 fluidSample(float3 p, device const float *field);
// Scalars deliberately avoid Swift/MSL float3 alignment differences.
// u: width height time level yaw pitch roll tiltX tiltY heat ambient wind,
// cupRGB liquidRGB cremaRGB saucerRGB steamRGB, retro pour dotsCount dither reserved.
float hash21(float2 p) { return fract(sin(dot(p,float2(127.1,311.7)))*43758.5453); }
float noise(float2 p) {
    float2 i=floor(p), f=fract(p); f=f*f*(3.-2.*f);
    return mix(mix(hash21(i),hash21(i+float2(1,0)),f.x),
               mix(hash21(i+float2(0,1)),hash21(i+1.),f.x),f.y);
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
float2 scene(float3 p, device const float *u, device const float *w, CupBody body) {
    float2 hit=crockery(p,body);
    float liquid=u[32]>.5 ? fluidDistance(p,w,body) :
        max(max((p.y-liquidHeight(p.xz,u,w,body))*.7,length(p.xz)-.745),.125-p.y);
    if(liquid<hit.x) hit=float2(liquid,2);
    if(p.y+.13<hit.x) hit=float2(p.y+.13,3);
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
float shadow(float3 p,float3 light,CupBody body) {
    float result=1., t=.025;
    for(int i=0;i<36;i++) {
        float d=crockery(p+light*t,body).x;
        if(d<.0008) return .08;
        result=min(result,12.*d/t); t+=clamp(d,.014f,.18f);
        if(t>3.)break;
    }
    return clamp(result,.08f,1.f);
}
float3 environment(float3 d, float ambient, device const float *u) {
    float3 sky=mix(float3(.10,.13,.19),float3(.35,.39,.46),smoothstep(-.2f,.9f,d.y));
    // Broad rectangular window, including two subtle mullions.
    float2 a=float2(d.x/max(.05f,d.y),d.z/max(.05f,d.y));
    float window=(1.-smoothstep(.37f,.42f,abs(a.x+.58)))*(1.-smoothstep(.5f,.56f,abs(a.y+.4)));
    float bars=1.-.38*exp(-abs(a.x+.58)*120.)-.28*exp(-abs(a.y+.4)*120.);
    float3 env=sky*.6+float3(1.4,1.18,.83)*window*bars*mix(.6f,1.f,ambient);
    if(u[34]>.5) {
        // Off-camera neon tubes have finite width, so reflections slide over
        // moving liquid and ceramic rather than being painted onto the image.
        float red=exp(-pow((d.x+.72)*17.,2.))*exp(-pow((d.y-.22)*2.7,4.));
        float teal=exp(-pow((d.z+.76)*22.,2.))*exp(-pow((d.y-.30)*3.,4.));
        float breath=.97+.03*sin(u[2]*.7);
        env+=float3(.9,.055,.03)*red*breath+float3(.012,.28,.25)*teal;
    }
    return env;
}
float3 material(float3 p, int mat, device const float *u, CupBody body) {
    if(mat==0)p=cupLocal(p,body);
    float3 cup=float3(u[12],u[13],u[14]), saucer=float3(u[21],u[22],u[23]);
    if(mat==0 || mat==1) {
        float3 col=mat==0?cup:saucer;
        if(u[27]>.5) {
            float stripe=mat==0 ? (1.-smoothstep(.010f,.014f,abs(p.y-.862)))*(1.-smoothstep(.84f,.87f,length(p.xz)))
                               : (1.-smoothstep(.010f,.017f,abs(length(p.xz)-1.14)));
            col=mix(col,float3(.27,.045,.055),stripe);
            col*=.975+.025*noise(p.xz*83.+p.y*13.);
        }
        return col;
    }
    if(mat==3) {
        float grain=noise(float2(p.x*1.8,p.z*32.))*.7+noise(float2(p.x*.5,p.z*95.))*.3;
        float seams=1.-.28*(1.-smoothstep(.004f,.012f,abs(fract(p.z*.7)-.5)));
        return mix(float3(.048,.032,.025),float3(.115,.075,.046),grain)*seams;
    }
    // APIC cream is exclusively advected material, never decorative noise.
    if(u[32]>.5)return float3(u[15],u[16],u[17]);
    float r=length(p.xz), a=atan2(p.z,p.x);
    float swirls=noise(float2(a*3.+u[2]*.13,r*19.+sin(a*3.+u[2]*.22)*.6));
    float cream=smoothstep(.58f,.735f,r)*(.40+.60*swirls);
    cream+=.15*smoothstep(.67f,.9f,swirls)*smoothstep(.3f,.6f,r);
    if(u[32]>.5)cream*=(1.-smoothstep(.72f,.79f,r))*smoothstep(.15f,.3f,p.y);
    return mix(float3(u[15],u[16],u[17])*.48,float3(u[18],u[19],u[20]),clamp(cream,0.f,.9f));
}
float headlightPhase(device const float *u) {return fmod(u[2]+4.,22.);}
float3 dinerLighting(float3 p,float3 n,float3 rd,device const float *u,CupBody body) {
    if(u[34]<.5)return float3(0);
    float3 redDir=normalize(float3(-1.5,.8,.4)),blueDir=normalize(float3(.4,.9,-1.6));
    float3 color=float3(.18,.012,.006)*max(0.f,dot(n,redDir));
    color+=float3(.005,.048,.043)*max(0.f,dot(n,blueDir));
    float phase=headlightPhase(u);
    if(phase<6.) {
        float travel=mix(-4.f,4.f,phase/6.);
        float fade=smoothstep(0.f,.9f,phase)*(1.-smoothstep(4.8f,6.f,phase));
        float3 lamp=float3(travel,1.7,-3.4);
        float3 l=normalize(lamp-p);
        // Two soft headlight beams move coherently over the entire diorama.
        float beam=exp(-pow((p.x-travel*.63-.15)*2.4,2.))+
                   .7*exp(-pow((p.x-travel*.63+.65)*2.4,2.));
        float diffuse=max(0.f,dot(n,l));
        float spec=pow(max(0.f,dot(n,normalize(l-rd))),90.f);
        color+=float3(.28,.23,.15)*fade*beam*(diffuse+.5*spec)*shadow(p+n*.009,l,body);
    }
    return color;
}
// One bounded refracted path through the actual reconstructed liquid. Coffee
// absorbs light; only added cream scatters it. No opaque brown surface coat.
float3 transmittedSolid(float3 p,int mat,float3 rd,device const float *u,CupBody body,
                        device const uint4 *film,device const uint4 *dry,device const float2 *waves) {
    float3 n=solidNormal(p,body),l=normalize(float3(-.65,1.3,-.7));
    float3 warm=mix(float3(1.05,.59,.32),float3(1.15,.97,.76),u[10]);
    float3 base=material(p,mat,u,body);
    float4 coating=0.;
    if((mat==1||mat==3)&&abs(p.y-filmBaseHeight(p.xz))<.025) {
        coating=sampleFilm(p.xz,film,dry);base=stainColor(base,coating);
        if(coating.x>.0001)n=wetNormal(p,n,film,waves);
    }
    float shade=shadow(p+n*.008,l,body);
    float3 col=base*(float3(.30,.29,.27)*(u[10]*.5+.5)+warm*max(0.f,dot(n,l))*shade)
         +base*dinerLighting(p,n,rd,u,body);
    float wet=smoothstep(.00008f,.0015f,coating.x);
    float f=.0204+.9796*pow(1.-max(0.f,dot(-rd,n)),5.);
    return mix(col,environment(reflect(rd,n),u[10],u),wet*f);
}
float3 coffeeTransmission(float3 p,float3 n,float3 rd,device const float *u,
                         device const float *field,CupBody body,
                         device const uint4 *film,device const uint4 *dry,device const float2 *waves) {
    float3 ray=refract(rd,n,1./1.333),q=p-n*.004;
    float3 transmission=1.,radiance=0.;bool entered=false,exited=false,bounced=false;
    // Theme color controls absorption, not an opaque diffuse paint layer.
    float3 opticalDensity=-log(clamp(float3(u[15],u[16],u[17]),.0001f,.9999f));
    float neutral=min(opticalDensity.x,min(opticalDensity.y,opticalDensity.z));
    float3 coffeeAbsorption=neutral*1.25+(opticalDensity-neutral)*3.5;
    float3 light=normalize(float3(-.65,1.3,-.7));
    float illumination=.30+.65*max(0.f,dot(n,light))*shadow(p+n*.008,light,body);
    for(int j=0;j<112;j++) {
        float2 ceramic=crockery(q,body);
        if(q.y+.13<ceramic.x)ceramic=float2(q.y+.13,3);
        if(ceramic.x<.0015)
            return radiance+transmission*transmittedSolid(q,int(ceramic.y),ray,u,body,film,dry,waves);
        float4 sample=fluidSample(q,field);
        if(!exited && sample.x<.0015) {
            entered=true;
            float step=min(.018f,max(.003f,ceramic.x*.8));
            float cream=clamp(sample.y,0.f,1.f);
            float3 absorb=coffeeAbsorption*(1.-cream);
            float3 scatter=float3(32.)*cream;
            float3 extinction=absorb+scatter+1e-5;
            float3 attenuation=exp(-extinction*step);
            radiance+=transmission*(1.-attenuation)*(scatter/extinction)
                     *float3(.92,.86,.72)*illumination;
            transmission*=attenuation;
            if(max(transmission.x,max(transmission.y,transmission.z))<.003)return radiance;
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
float3 shadeHit(float3 p,float3 rd,int mat, device const float *u,device const float *w, CupBody body,
                device const uint4 *film,device const uint4 *dry,device const float2 *waves,
                device const float4 *impacts) {
    float3 n=normalAt(p,u,w,body), light=normalize(float3(-.65,1.3,-.7));
    if(mat==2 && u[32]>.5) {
        // Wider symmetric derivatives smooth grid-scale normals without
        // smoothing particle motion or replacing the free liquid surface.
        float e=.010;
        n=normalize(float3(fluidDistance(p+float3(e,0,0),w,body)-fluidDistance(p-float3(e,0,0),w,body),
                           fluidDistance(p+float3(0,e,0),w,body)-fluidDistance(p-float3(0,e,0),w,body),
                           fluidDistance(p+float3(0,0,e),w,body)-fluidDistance(p-float3(0,0,e),w,body)));
    }
    float4 coating=0.;float wet=0.;
    if(u[32]>.5 && (mat==1||mat==3) && abs(p.y-filmBaseHeight(p.xz))<.025) {
        coating=sampleFilm(p.xz,film,dry);wet=smoothstep(.00008f,.0015f,coating.x);
        if(wet>0.)n=normalize(mix(n,wetNormal(p,n,film,waves),wet));
    }
    if(mat==2 && u[32]>.5 && inCup(p,body)) {
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
    float3 base=stainColor(material(p,mat,u,body),coating);
    float milk=0.;
    if(mat==2 && u[32]>.5) {
        milk=clamp(fluidSample(p-n*.008,w).y,0.f,1.f);
        base=mix(base,float3(.84,.76,.62),milk);
    }
    float diffuse=max(0.f,dot(n,light))*shadow(p+n*.006,light,body);
    float ao=1.;
    for(int i=1;i<=4;i++) { float d=.045*i; ao-=max(0.f,d-crockery(p+n*d,body).x)*(.7/i); }
    ao=clamp(ao,.3f,1.f);
    float room=u[10];
    float3 warm=mix(float3(1.05,.59,.32),float3(1.15,.97,.76),room);
    float3 fill=float3(.30,.29,.27)*(room*.5+.5);
    fill+=float3(.12,.13,.15)*max(0.f,dot(n,normalize(float3(.2,.6,1.))));
    float3 col=base*(fill+warm*diffuse)*ao;
    col+=base*dinerLighting(p,n,rd,u,body);
    if(mat==2 && u[32]>.5)col=coffeeTransmission(p,n,rd,u,w,body,film,dry,waves);
    float3 refl=reflect(rd,n);
    float fres=pow(1.-max(0.f,dot(-rd,n)),5.);
    if(mat!=3 || wet>0.) {
        float3 env=environment(refl,room,u);
        if(mat==2) {
            // Reflect the inner ceramic wall, not just a painted highlight.
            float t=.025;
            for(int i=0;i<24;i++) {
                float3 rp=p+n*.009+refl*t; float2 h=crockery(rp,body);
                if(h.x<.004) {env=material(rp,int(h.y),u,body)*(.3+.35*max(0.f,rp.y));break;}
                t+=max(.008f,h.x*.85); if(t>2.)break;
            }
        }
        float strength=mat==2 ? .0204+.9796*fres : (mat==3?0.:.045+.32*fres);
        strength=mix(strength,.0204+.9796*fres,wet);
        col=mix(col,env,strength);
        float spec=pow(max(0.f,dot(n,normalize(light-rd))),mat==2?mix(220.f,100.f,milk):mix(85.f,240.f,wet));
        col+=warm*spec*(mat==2?.9:.38)*shadow(p+n*.009,light,body);
    }
    // A slight atmospheric falloff avoids an infinite bright tabletop.
    if(mat==3) col=mix(col,float3(.025,.029,.034),smoothstep(2.f,8.f,length(p.xz)));
    return col;
}
float3 shadeSpray(float3 p,float3 rd,SprayHit hit,device const float *u,device const float *field,
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
        if(h.x<.002) {background=shadeHit(q,ray,int(h.y),u,field,body,film,dry,waves,impacts);break;}
        t+=max(.003f,h.x*.75);if(t>5.)break;
    }
    float3 density=-log(clamp(float3(u[15],u[16],u[17]),.0001f,.9999f));
    float neutral=min(density.x,min(density.y,density.z));
    float3 absorb=(neutral*1.25+(density-neutral)*3.5)*(1.-hit.cream),scatter=32.*hit.cream;
    float3 extinction=absorb+scatter+1e-5,transmission=exp(-extinction*thickness);
    float3 col=background*transmission+(1.-transmission)*(scatter/extinction)*float3(.60,.55,.45);
    float fres=.0204+.9796*pow(1.-max(0.f,dot(-rd,n)),5.);
    col=mix(col,environment(reflect(rd,n),u[10],u),fres);
    float3 light=normalize(float3(-.65,1.3,-.7));
    col+=float3(1.1,.95,.8)*pow(max(0.f,dot(n,normalize(light-rd))),240.f)*.6;
    return col;
}
float steamDensity(float3 p,device const float *u,CupBody body) {
    float height=p.y-(u[32]>.5?u[33]:.30+.62*u[3]);
    if(height<0. || height>1.85)return 0.;
    float2 center=-slopes(u)*height+float2(u[11]*.014*height,0)+(u[32]>.5?body.position.xz:float2(0));
    float2 q=p.xz-center;
    float density=0., tm=u[2];
    for(int i=0;i<3;i++) {
        float k=float(i), phase=height*5.-tm*(1.1+u[9]*.4)+k*2.1;
        float2 path=float2(sin(phase)*(.08+height*.05)+sin(k*2.4)*.21,
                           cos(phase*.8)*.08+cos(k*2.4)*.18);
        float radius=.035+height*.045;
        density+=exp(-dot(q-path,q-path)/(radius*radius));
    }
    float envelope=smoothstep(0.f,.17f,height)*(1.-smoothstep(.7f,1.85f,height));
    return density*envelope*(.13+u[9]*.1)*u[3];
}
kernel void coffee(device uchar *out [[buffer(0)]], device const float *u [[buffer(1)]],
                   device const float *w [[buffer(2)]],device const float4 *dots [[buffer(3)]],
                   device const CupBody &body [[buffer(4)]],
                   device const FluidParticle *fluidParticles [[buffer(5)]],
                   device const int *heads [[buffer(6)]],device const int *next [[buffer(7)]],
                   device const uint *active [[buffer(8)]],device const uint4 *film [[buffer(9)]],
                   device const uint4 *dry [[buffer(10)]],device const float2 *waves [[buffer(11)]],
                   device const float4 *impacts [[buffer(12)]],
                   uint2 gid [[thread_position_in_grid]]) {
    uint width=uint(u[0]),height=uint(u[1]); if(gid.x>=width||gid.y>=height)return;
    float2 uv=(float2(gid)+.5)/float2(width,height);
    float yaw=u[4],pitch=u[5],roll=u[6];
    float3 target=float3(0,.55,0);
    if(u[32]>.5)target.xz=clamp(body.position.xz*.45,float2(-.85),float2(.85));
    float upY=1.-2.*(body.rotation.x*body.rotation.x+body.rotation.z*body.rotation.z);
    float distance=3.4+(u[32]>.5?.8*(1.-abs(upY))+.15*min(1.f,length(body.position.xz)):0.);
    float3 ro=target+distance*float3(cos(pitch)*sin(yaw),sin(pitch),cos(pitch)*cos(yaw));
    float3 forward=normalize(target-ro),right=normalize(cross(forward,float3(0,1,0))),up=cross(right,forward);
    float3 rr=right*cos(roll)+up*sin(roll),ru=up*cos(roll)-right*sin(roll);
    float2 screen=float2((uv.x*2.-1.)*float(width)/height*.95,(1.-uv.y*2.)*.95-.28);
    float3 rd=normalize(forward*1.9+rr*screen.x+ru*screen.y);
    float t=0.,mat=-1.;
    for(int i=0;i<180;i++) {
        float2 hit=scene(ro+rd*t,u,w,body);
        if(hit.x<.0015+t*.0003){mat=hit.y;break;}
        t+=max(.0008f,hit.x*.78);if(t>14.)break;
    }
    float3 col=float3(.026,.030,.036);
    SprayHit spray;spray.radius=0.;
    if(u[32]>.5)spray=traceSpray(ro,rd,t,fluidParticles,heads,next,active);
    if(spray.radius>0.) {t=spray.distance;col=shadeSpray(ro+rd*t,rd,spray,u,w,body,film,dry,waves,impacts);}
    else if(mat>=0.)col=shadeHit(ro+rd*t,rd,int(mat),u,w,body,film,dry,waves,impacts);
    // Integrate wisps only inside their bounding volume, in front of solids.
    float3 center=float3(body.position.x,1.65,body.position.z);float b=dot(ro-center,rd);
    float disc=b*b-dot(ro-center,ro-center)+1.45*1.45;
    if(disc>0.) {
        float start=max(0.f,-b-sqrt(disc)),end=min(t,-b+sqrt(disc));
        float step=.045;
        for(float d=start;d<end;d+=step) {
            float3 p=ro+rd*d;
            float density=steamDensity(p,u,body)*step*4.;
            float shaft=exp(-pow((p.x+.45-p.y*.35)*3.,2.));
            float3 steamColor=float3(u[24],u[25],u[26])*(.5+.2*u[10]);
            if(u[34]>.5)steamColor*=float3(1.12,.96,.78)*(.65+shaft*.8);
            col=mix(col,steamColor,clamp(density,0.f,1.f));
        }
    }
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
    for(int i=0;i<int(u[29]);i++) {
        float4 p=dots[i];float2 delta=(uv-p.xy)*float2(width,height);
        float a=exp(-dot(delta,delta)/(p.z*p.z))*p.w;
        col=mix(col,float3(u[24],u[25],u[26]),min(.6f,a));
    }
    col*=1.-(u[34]>.5?.32:.20)*dot(uv-.5,uv-.5);
    col=pow(clamp(col,0.f,1.f),float3(1./2.2));
    // Fine static ordered dither: retro texture without temporal noise/flicker.
    if(u[30]>.5) {
        constexpr float bayer[16]={0,8,2,10,12,4,14,6,3,11,1,9,15,7,13,5};
        float d=(bayer[(gid.y%4)*4+gid.x%4]+.5)/16.;
        col=floor(col*63.+d)/63.;
    }
    uint index=(gid.y*width+gid.x)*3;
    out[index]=uchar(clamp(col.r*255.,0.f,255.f));
    out[index+1]=uchar(clamp(col.g*255.,0.f,255.f));
    out[index+2]=uchar(clamp(col.b*255.,0.f,255.f));
}
