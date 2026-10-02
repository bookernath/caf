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
float3 keyDir() {return normalize(float3(-.65,1.3,-.7));}
// Warm key and cool sky ambient: shade is modelled by temperature, not gray fill.
float3 keyLight(float room) {return mix(float3(1.05,.59,.32),float3(1.15,.97,.76),room)*1.35;}
float3 ambientLight(float3 n,float room) {
    float3 a=mix(float3(.07,.06,.05),float3(.17,.20,.26),smoothstep(-.6f,.9f,n.y));
    a+=float3(.07,.075,.085)*max(0.f,dot(n,normalize(float3(.2,.6,1.))));
    return a*(room*.6+.6);
}
// Soft key-light occlusion; 0 is a full umbra, the ambient term lights it.
float shadow(float3 p,float3 light,CupBody body) {
    float result=1., t=.025;
    for(int i=0;i<36;i++) {
        float d=crockery(p+light*t,body).x;
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
        float d=crockery(q,body).x, f=fluidSample(q,field).x;
        if(d<.0008) return 0.;
        if(f<.002)liquid=.45;
        result=min(result,12.*d/t); t+=clamp(min(d,max(f,.004f)),.006f,.18f);
        if(t>3.)break;
    }
    return smoothstep(0.f,1.f,clamp(result,0.f,1.f))*liquid;
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
float fresnelRough(float nv,float f0,float rough) {
    return f0+(max(1.-rough,f0)-f0)*pow(1.-clamp(nv,0.f,1.f),5.f);
}
// Value noise averaged toward its mean once a cell is under ~2 pixels wide.
float filteredNoise(float2 p,float footprint) {
    return mix(noise(p),.5,smoothstep(.5f,1.5f,footprint));
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
        return col;
    }
    if(mat==3) {
        // Grain is anisotropic: its z frequency dominates the footprint test.
        float grain=filteredNoise(float2(p.x*1.8,p.z*32.),fw*32.)*.7+filteredNoise(float2(p.x*.5,p.z*95.),fw*95.)*.3;
        float seamWidth=max(.008f,fw*.7);
        float seams=1.-.28*(1.-smoothstep(seamWidth*.5,seamWidth*1.5,abs(fract(p.z*.7)-.5)))*(.008/seamWidth);
        return mix(float3(.048,.032,.025),float3(.115,.075,.046),grain)*seams;
    }
    // APIC cream is exclusively advected material, never decorative noise.
    if(u[32]>.5)return float3(u[15],u[16],u[17]);
    float r=length(p.xz), a=atan2(p.z,p.x);
    float swirls=noise(float2(a*3.+u[2]*.13,r*19.+sin(a*3.+u[2]*.22)*.6));
    float cream=smoothstep(.58f,.735f,r)*(.40+.60*swirls);
    cream+=.15*smoothstep(.67f,.9f,swirls)*smoothstep(.3f,.6f,r);
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
        float spec=ggxSpecular(n,-rd,l,.25,.04);
        color+=float3(.28,.23,.15)*fade*beam*(diffuse+spec)*shadow(p+n*.009,l,body);
    }
    return color;
}
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
    float3 col=base*(ambientLight(n,u[10])+keyLight(u[10])*max(0.f,dot(n,l))*shade)
         +base*dinerLighting(p,n,rd,u,body);
    float wet=smoothstep(.00008f,.0015f,coating.x);
    return mix(col,environment(reflect(rd,n),u[10],u),wet*fresnelRough(dot(-rd,n),.02,.1));
}
float3 coffeeTransmission(float3 p,float3 n,float3 rd,float fw,float keyShade,device const float *u,
                         device const float *field,CupBody body,
                         device const uint4 *film,device const uint4 *dry,device const float2 *waves) {
    float3 ray=refract(rd,n,1./1.333),q=p-n*.004;
    float3 transmission=1.,radiance=0.;bool entered=false,exited=false,bounced=false;
    // Theme color controls absorption, not an opaque diffuse paint layer.
    float3 opticalDensity=-log(clamp(float3(u[15],u[16],u[17]),.0001f,.9999f));
    float neutral=min(opticalDensity.x,min(opticalDensity.y,opticalDensity.z));
    float3 coffeeAbsorption=neutral*1.25+(opticalDensity-neutral)*3.5;
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
// The free surface comes from the particle field alone: fluidDistance also
// folds in the ceramic, so its differences straddle the wall into speckles.
// Hits clipped by the ceramic sit just off its face, so fall back to it.
float3 liquidNormal(float3 p,device const float *field,CupBody body) {
    float3 g=fluidGradient(p,field);
    return length(g)>1e-3 ? normalize(g) : solidNormal(p,body);
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
    float fw=fp/max(.35f,abs(dot(rd,n)));
    float3 base=stainColor(material(p,mat,fw,u,body),coating);
    float milk=0.;
    if(mat==2 && real) {
        milk=clamp(fluidSample(p-n*.008,w).y,0.f,1.f);
        base=mix(base,float3(.84,.76,.62),milk);
    }
    // One key-light shadow ray shared by diffuse, specular and in-scattering.
    float key=real&&(mat==1||mat==3) ? shadowSpill(p+n*.006,light,body,w) : shadow(p+n*.006,light,body);
    float ao=1.;
    for(int i=1;i<=4;i++) { float d=.045*i; ao-=max(0.f,d-crockery(p+n*d,body).x)*(.7/i); }
    ao=clamp(ao,.3f,1.f);
    float room=u[10];
    float3 warm=keyLight(room);
    float3 col=base*(ambientLight(n,room)*ao+warm*max(0.f,dot(n,light))*key);
    col+=base*dinerLighting(p,n,rd,u,body);
    if(mat==2 && real)col=coffeeTransmission(p,n,rd,fw,key,u,w,body,film,dry,waves);
    if(mat!=3 || wet>0.) {
        float3 refl=reflect(rd,n);
        float3 env=environment(refl,room,u);
        if(mat==2) {
            // Reflect the inner ceramic wall, not just a painted highlight.
            float t=.025;
            for(int i=0;i<24;i++) {
                float3 rp=p+n*.009+refl*t; float2 h=crockery(rp,body);
                if(h.x<.004) {env=material(rp,int(h.y),fw,u,body)*(.3+.35*max(0.f,rp.y));break;}
                t+=max(.008f,h.x*.85); if(t>2.)break;
            }
        }
        // Liquid and water films: F0 .02, smooth. Glaze: F0 .04 and rougher,
        // so its grazing reflection saturates below 1. Diffuse gets (1-F).
        float f0=mat==2 ? .02 : mix(.04f,.02f,wet);
        float rough=mat==2 ? mix(.08f,.16f,milk) : mix(.34f,.09f,wet);
        float strength=fresnelRough(dot(v,n),f0,mat==2?0.:rough)*(mat==3?wet:1.);
        col=mix(col,env,strength);
        col+=warm*ggxSpecular(n,v,light,rough,f0)*key*(mat==3?wet:1.);
    }
    // A slight atmospheric falloff avoids an infinite bright tabletop.
    if(mat==3) col=mix(col,float3(.025,.029,.034),smoothstep(2.f,8.f,length(p.xz)));
    return col;
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
    float3 density=-log(clamp(float3(u[15],u[16],u[17]),.0001f,.9999f));
    float neutral=min(density.x,min(density.y,density.z));
    float3 absorb=(neutral*1.25+(density-neutral)*3.5)*(1.-hit.cream),scatter=32.*hit.cream;
    float3 extinction=absorb+scatter+1e-5,transmission=exp(-extinction*thickness);
    float3 col=background*transmission+(1.-transmission)*(scatter/extinction)*float3(.60,.55,.45);
    col=mix(col,environment(reflect(rd,n),u[10],u),fresnelRough(dot(-rd,n),.02,0.));
    col+=keyLight(u[10])*ggxSpecular(n,-rd,keyDir(),.07,.02);
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
float interleavedGradientNoise(float2 p) {return fract(52.9829189*fract(dot(p,float2(.06711056,.00583715))));}
float3 traceScene(float2 pixel,float stillFrame,device const float *u,device const float *w,CupBody body,
                  device const FluidParticle *fluidParticles,device const int *heads,device const int *next,
                  device const uint *active,device const uint4 *film,device const uint4 *dry,
                  device const float2 *waves,device const float4 *impacts,thread float2 &info) {
    float width=u[0],height=u[1];
    float2 uv=pixel/float2(width,height);
    float yaw=u[4],pitch=u[5],roll=u[6];
    float3 target=float3(0,.55,0);
    if(u[32]>.5)target.xz=clamp(body.position.xz*.45,float2(-.85),float2(.85));
    float upY=1.-2.*(body.rotation.x*body.rotation.x+body.rotation.z*body.rotation.z);
    float distance=3.4+(u[32]>.5?.8*(1.-abs(upY))+.15*min(1.f,length(body.position.xz)):0.);
    float3 ro=target+distance*float3(cos(pitch)*sin(yaw),sin(pitch),cos(pitch)*cos(yaw));
    float3 forward=normalize(target-ro),right=normalize(cross(forward,float3(0,1,0))),up=cross(right,forward);
    float3 rr=right*cos(roll)+up*sin(roll),ru=up*cos(roll)-right*sin(roll);
    float2 screen=float2((uv.x*2.-1.)*width/height*.95,(1.-uv.y*2.)*.95-.28);
    float3 rd=normalize(forward*1.9+rr*screen.x+ru*screen.y);
    float t=0.,mat=-1.;
    for(int i=0;i<180;i++) {
        float2 hit=scene(ro+rd*t,u,w,body);
        if(hit.x<.0015+t*.0003){mat=hit.y;break;}
        t+=max(.0008f,hit.x*.78);if(t>14.)break;
    }
    // One pixel subtends 1.9/height screen units at focal length 1.9.
    float pixelAngle=1./height;
    float3 col=float3(.026,.030,.036);
    SprayHit spray;spray.radius=0.;
    if(u[32]>.5)spray=traceSpray(ro,rd,t,fluidParticles,heads,next,active);
    if(spray.radius>0.) {t=spray.distance;mat=4.;col=shadeSpray(ro+rd*t,rd,spray,t*pixelAngle,u,w,body,film,dry,waves,impacts);}
    else if(mat>=0.)col=shadeHit(ro+rd*t,rd,int(mat),t*pixelAngle,u,w,body,film,dry,waves,impacts);
    else t=14.;
    info=float2(t,mat);
    // Steam: front-to-back Beer-Lambert from a per-pixel jittered start (the
    // temporal pass averages it) plus one light-ward sample for self-shadowing.
    float3 center=float3(body.position.x,1.65,body.position.z);float b=dot(ro-center,rd);
    float disc=b*b-dot(ro-center,ro-center)+1.45*1.45;
    if(disc>0.) {
        float start=max(0.f,-b-sqrt(disc)),end=min(t,-b+sqrt(disc));
        float step=.045;
        float3 light=keyDir(),transmittance=1.,scattered=0.;
        float3 steamColor=float3(u[24],u[25],u[26])*(.5+.2*u[10]);
        start+=step*interleavedGradientNoise(floor(pixel)+5.588238*fmod(stillFrame,64.f));
        for(float d=start;d<end;d+=step) {
            float3 p=ro+rd*d;
            float sigma=steamDensity(p,u,body)*4.;
            if(sigma<1e-4)continue;
            float self=exp(-steamDensity(p+light*.12,u,body)*4.*.12*3.);
            float3 lit=steamColor*(.55+.6*self);
            if(u[34]>.5)lit*=float3(1.12,.96,.78)*(.65+exp(-pow((p.x+.45-p.y*.35)*3.,2.))*.8);
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
