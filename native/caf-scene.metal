#include <metal_stdlib>
using namespace metal;

// A full source particle carries eight integer micro-volumes. Lifecycle is
// separate from weight so splitting, sipping, escape and film transfer compose.
struct FluidParticle { float4 x, v, cx, cy, cz; };
constant uint MICRO = 8;
constant uint SLOT_CAPACITY = 120000;
constant int FILM_N = 384;
constant int FILM_CELLS = FILM_N*FILM_N;
constant float FILM_DX = 8./FILM_N;
constant float FILM_ORIGIN = -4.;
constant uint FILM_Q = 4096;
constant float PARTICLE_VOLUME = .0425*.0425*.0425;
constant float FILM_VOLUME_Q = PARTICLE_VOLUME/(8.*4096.);
struct SprayHit { float distance; float3 normal; float cream; float radius; };
bool particleAlive(FluidParticle p) { return p.x.w>0. && p.cz.w==0.; }
float particleWeight(FluidParticle p) { return p.x.w/8.; }
float particleRadius(FluidParticle p) { return pow(3.*PARTICLE_VOLUME*max(.125f,particleWeight(p))/(4.*M_PI_F),1./3.); }
int2 filmCoord(float2 p) { return int2(floor((p-FILM_ORIGIN)/FILM_DX)); }
bool inFilm(int2 c) {return all(c>=0)&&all(c<FILM_N);}
int filmIndex(int2 c) {return c.y*FILM_N+c.x;}
float2 filmCenter(int2 c) {return FILM_ORIGIN+(float2(c)+.5)*FILM_DX;}
float filmBaseHeight(float2 xz) {
    float r=length(xz);
    // Upper envelope of the two saucer capsules; everything else is table.
    float a=r<=.90 ? -.048 : -.085+sqrt(max(0.f,.037*.037-(r-.90)*(r-.90)));
    if(r>.937)a=-.13;
    float b=-.13;
    float slope=.09/.37;
    float t=(r-.90+slope*(.037/sqrt(1.+slope*slope)))/.37;
    if(t>=0. && t<=1.)b=-.085+slope*(r-.90)+.037*sqrt(1.+slope*slope);
    if(abs(r-1.27)<.037)b=max(b,.005+sqrt(.037*.037-(r-1.27)*(r-1.27)));
    return max(-.13f,max(a,b));
}
float filmDepth(uint q) {return float(q)*FILM_VOLUME_Q/(FILM_DX*FILM_DX);}


// Shared by rendering and physics. Local cup coordinates never change; its
// center-of-mass transform moves the SAME SDF used for particle/grid contacts.
// float4-only layout is also the Swift host's 80-byte diagnostic state.
struct CupBody {
    float4 position;     // xyz center of mass; w = dynamics enabled
    float4 rotation;     // unit quaternion, xyzw
    float4 velocity;
    float4 angular;
    float4 contact;      // impact speed, sliding speed, penetration, reserved
};
constant float3 CUP_COM = float3(0,.52,0);
float3 rotateQ(float4 q, float3 p) {
    return p + 2.*cross(q.xyz,cross(q.xyz,p)+q.w*p);
}
float4 multiplyQ(float4 a,float4 b) {
    return float4(a.w*b.xyz+b.w*a.xyz+cross(a.xyz,b.xyz),a.w*b.w-dot(a.xyz,b.xyz));
}
float3 cupLocal(float3 p, CupBody b) {
    return rotateQ(float4(-b.rotation.xyz,b.rotation.w),p-b.position.xyz)+CUP_COM;
}
float3 cupWorld(float3 p, CupBody b) {
    return rotateQ(b.rotation,p-CUP_COM)+b.position.xyz;
}
float3 cupVelocity(float3 p,CupBody b) {
    return b.position.w>.5 ? b.velocity.xyz+cross(b.angular.xyz,p-b.position.xyz) : float3(0);
}
float segment(float2 p, float2 a, float2 b) {
    float2 v=b-a;return length(p-a-v*clamp(dot(p-a,v)/dot(v,v),0.f,1.f));
}
float cupShape(float3 p) {
    float r=length(p.xz);float2 q=float2(r,p.y);
    float d=segment(q,float2(0,.07),float2(.52,.07));
    d=min(d,segment(q,float2(.52,.07),float2(.66,.13)));
    d=min(d,segment(q,float2(.66,.13),float2(.75,.27)));
    d=min(d,segment(q,float2(.75,.27),float2(.80,1.)))-.048;
    d=min(d,length(float2(r-.48,p.y+.003))-.045);
    d=min(d,length(float2(r-.8,p.y-1.))-.054);
    float2 h=float2((p.x-1.0)*.92,(p.y-.58)*1.15);
    return min(d,max(length(float2(length(h)-.31,p.z))-.065,.78-r));
}
float saucerShape(float3 p) {
    float2 q=float2(length(p.xz),p.y);
    return min(segment(q,float2(0,-.085),float2(.90,-.085)),
               segment(q,float2(.90,-.085),float2(1.27,.005)))-.037;
}
float fixedSolid(float3 p) {return min(saucerShape(p),p.y+.13);}
float3 fixedNormal(float3 p) {
    float e=.001;
    float3 n=float3(fixedSolid(p+float3(e,0,0))-fixedSolid(p-float3(e,0,0)),
                    fixedSolid(p+float3(0,e,0))-fixedSolid(p-float3(0,e,0)),
                    fixedSolid(p+float3(0,0,e))-fixedSolid(p-float3(0,0,e)));
    return n/max(length(n),1e-8f);
}
float2 crockery(float3 p,CupBody b) {
    float cup=cupShape(cupLocal(p,b)),saucer=saucerShape(p);
    return cup<saucer?float2(cup,0):float2(saucer,1);
}
float solidDistance(float3 p,CupBody b) {return min(crockery(p,b).x,p.y+.13);}
float3 solidNormal(float3 p,CupBody b) {
    float e=.001;
    float3 n=float3(solidDistance(p+float3(e,0,0),b)-solidDistance(p-float3(e,0,0),b),
                    solidDistance(p+float3(0,e,0),b)-solidDistance(p-float3(0,e,0),b),
                    solidDistance(p+float3(0,0,e),b)-solidDistance(p-float3(0,0,e),b));
    return n/max(length(n),1e-8f);
}
bool isCupContact(float3 p,CupBody b) {
    return cupShape(cupLocal(p,b))<fixedSolid(p);
}
bool inCup(float3 world,CupBody b) {
    float3 p=cupLocal(world,b);
    float outer=p.y<.27 ? .66+(p.y-.13)/.14*.09 : .75+(p.y-.27)/.73*.05;
    return p.y>.125 && p.y<1.02 && length(p.xz)<outer-.045;
}
// Shell support samples include both sides of the rolled foot/rim and handle.
// A small contact skin covers the gap between neighboring angular samples.
constant int CUP_CONTACTS=240;
float3 cupSupport(int i) {
    if(i<192) {
        int ring=i/32;float a=float(i%32)*6.283185307/32.;
        float r=ring==0?.48:(ring==1?.525:(ring==2?.70:(ring==3?.794:(ring==4?.854:.80))));
        float y=ring==0?-.048:(ring==1?-.003:(ring==2?.095:(ring==3?.27:(ring==4?1.:1.054))));
        return float3(cos(a)*r,y,sin(a)*r);
    }
    // Exposed outer edge and front/back of the handle's elliptical torus.
    int k=i-192;float a=float(k%16)*6.283185307/16.;
    float r=k<16?.375:.31;
    return float3(1.+cos(a)*r/.92,.58+sin(a)*r/1.15,k<16?0.:(k<32?.065:-.065));
}
float3 inverseInertia(float3 torque,CupBody b) {
    float3 local=rotateQ(float4(-b.rotation.xyz,b.rotation.w),torque);
    return rotateQ(b.rotation,local*float3(3.7,2.7,3.7));
}
void addBodyImpulse(thread CupBody &b,float3 impulse,float3 r) {
    b.velocity.xyz+=impulse; // cup mass = 1; fluid full mass = .25
    b.angular.xyz+=inverseInertia(cross(r,impulse),b);
}
void accumulateReaction(device atomic_int *reaction,float3 impulse,float3 arm) {
    float3 torque=cross(arm,impulse);
    for(int a=0;a<3;a++) {
        atomic_fetch_add_explicit(reaction+a,int(clamp(impulse[a],-.5f,.5f)*1e6),memory_order_relaxed);
        atomic_fetch_add_explicit(reaction+3+a,int(clamp(torque[a],-.5f,.5f)*1e6),memory_order_relaxed);
    }
}

kernel void cupEvents(device CupBody &state [[buffer(14)]],device const float *s [[buffer(10)]],
                      uint i [[thread_position_in_grid]]) {
    if(i)return;
    state.position.w=s[15];
    if(s[15]<.5){state.velocity=0.;state.angular=0.;return;}
    CupBody b=state;
    if(s[14]!=0.) {
        float3 impulse=float3(cos(s[7]),0,-sin(s[7]))*s[14];
        addBodyImpulse(b,impulse,float3(0,.48,0));
    }
    state=b;
}
kernel void cupStep(device CupBody &state [[buffer(14)]],device atomic_int *reaction [[buffer(15)]],
                    device const float *s [[buffer(10)]],device atomic_uint *stats [[buffer(12)]],
                    uint i [[thread_position_in_grid]]) {
    if(i)return;
    CupBody b=state;
    float3 linear=0.,torque=0.;
    for(int a=0;a<3;a++) {
        linear[a]=float(atomic_exchange_explicit(reaction+a,0,memory_order_relaxed))/1e6;
        torque[a]=float(atomic_exchange_explicit(reaction+3+a,0,memory_order_relaxed))/1e6;
    }
    b.contact=0.;
    if(s[15]<.5){state=b;return;}
    float dt=s[0];
    b.velocity.xyz+=clamp(linear,-.3f,.3f)+float3(s[1],s[2],s[3])*dt;
    b.angular.xyz+=inverseInertia(clamp(torque,-.2f,.2f),b);
    b.angular.xyz*=exp(-dt*.14);
    float speed=length(b.velocity.xyz);if(speed>7.75)b.velocity.xyz*=7.75/speed;
    float spin=length(b.angular.xyz);if(spin>17.72)b.angular.xyz*=17.72/spin;
    float3 oldPos=b.position.xyz;float4 oldQ=b.rotation;
    float3 beforeV=b.velocity.xyz,beforeW=b.angular.xyz;
    float3 g=float3(s[1],s[2],s[3]);
    float3 localW=rotateQ(float4(-b.rotation.xyz,b.rotation.w),beforeW);
    float initialEnergy=.5*dot(beforeV-g*dt,beforeV-g*dt)+.5*dot(localW,localW/float3(3.7,2.7,3.7));
    b.position.xyz+=b.velocity.xyz*dt;
    b.rotation=normalize(b.rotation+.5*multiplyQ(float4(b.angular.xyz*dt,0),b.rotation));
    // Non-penetration constraints on the actual shell, not its solid convex
    // hull: the handle can catch the saucer rather than filling the cup hole.
    for(int pass=0;pass<4;pass++)for(int k=0;k<CUP_CONTACTS;k++) {
        int j=(pass&1)?CUP_CONTACTS-1-k:k;
        float3 local=cupSupport(j);
        if(j>=192 && length(local.xz)<.78)continue;
        float3 p=cupWorld(local,b);float d=fixedSolid(p);
        if(d>=.001)continue;
        float3 n=fixedNormal(p),r=p-b.position.xyz;
        float3 angularGradient=cross(r,n);
        float invMass=1.+dot(angularGradient,inverseInertia(angularGradient,b));
        float lambda=min(.025f,.001-d)/max(1.f,invMass);
        b.position.xyz+=n*lambda;
        float3 da=inverseInertia(angularGradient*lambda,b);
        b.rotation=normalize(b.rotation+.5*multiplyQ(float4(da,0),b.rotation));
    }
    // Constraint displacement defines the supported velocity. Velocity-level
    // contact friction and modest restitution then handle sliding/impacts.
    b.velocity.xyz=(b.position.xyz-oldPos)/dt;
    float4 dq=multiplyQ(b.rotation,float4(-oldQ.xyz,oldQ.w));
    b.angular.xyz=dq.xyz*(dq.w<0.?-2.:2.)/dt;
    int contacts=0;
    for(int j=0;j<CUP_CONTACTS;j++) {
        float3 local=cupSupport(j);
        if(j>=192 && length(local.xz)<.78)continue;
        if(fixedSolid(cupWorld(local,b))<.004)contacts++;
    }
    for(int pass=0;pass<2;pass++)for(int k=0;k<CUP_CONTACTS;k++) {
        int j=(pass&1)?CUP_CONTACTS-1-k:k;
        float3 local=cupSupport(j);
        if(j>=192 && length(local.xz)<.78)continue;
        float3 p=cupWorld(local,b);float d=fixedSolid(p);
        b.contact.z=max(b.contact.z,max(0.f,-d));
        if(d>=.004)continue;
        float3 n=fixedNormal(p),r=p-b.position.xyz;
        float oldVN=dot(beforeV+cross(beforeW,r),n);
        float3 v=b.velocity.xyz+cross(b.angular.xyz,r);
        float vn=dot(v,n);
        float3 rn=cross(r,n);
        float desired=oldVN<-.6 ? -oldVN*.09 : 0.;
        float jn=max(0.f,desired-vn)/(1.+dot(rn,inverseInertia(rn,b)));
        addBodyImpulse(b,n*jn,r);
        float3 tangent=v-n*vn;float vt=length(tangent);
        if(vt>1e-6) {
            float3 direction=tangent/vt,rt=cross(r,direction);
            float jt=vt/(1.+dot(rt,inverseInertia(rt,b)));
            float normalBudget=jn+length(float3(s[1],s[2],s[3]))*dt/max(1,contacts)*.5;
            addBodyImpulse(b,-direction*min(jt,normalBudget*.72),r);
            b.contact.y=max(b.contact.y,vt);
        }
        b.contact.x=max(b.contact.x,max(0.f,-oldVN-.3));
    }
    if(contacts>0)b.angular.xyz*=exp(-dt*.65); // rolling resistance on glazed ceramic/wood
    // Sequential finite-shell contacts must not manufacture energy when many
    // support points become active together (notably an upside-down rim).
    localW=rotateQ(float4(-b.rotation.xyz,b.rotation.w),b.angular.xyz);
    float energy=.5*dot(b.velocity.xyz,b.velocity.xyz)+.5*dot(localW,localW/float3(3.7,2.7,3.7));
    float budget=max(0.f,initialEnergy+dot(g,b.position.xyz-oldPos));
    if(energy>budget && energy>1e-8) {
        float scale=sqrt(budget/energy);b.velocity.xyz*=scale;b.angular.xyz*=scale;
    }
    if(!all(isfinite(b.position))||!all(isfinite(b.rotation))||!all(isfinite(b.velocity))||!all(isfinite(b.angular))) {
        atomic_fetch_add_explicit(stats+24,1,memory_order_relaxed);
        // Contain a numerical failure; expose it, never silently reset liquid.
        b=state;b.velocity=0.;b.angular=0.;
    }
    atomic_fetch_max_explicit(stats+20,uint(b.contact.x*10000.),memory_order_relaxed);
    atomic_fetch_max_explicit(stats+21,uint(b.contact.z*1e6),memory_order_relaxed);
    state=b;
}

void depositFilm(thread FluidParticle &p,float impact,device const uint4 *film,
                 device atomic_uint *pending,device atomic_uint *details);
SprayHit traceSpray(float3 ro,float3 rd,float maxT,device const FluidParticle *particles,
                    device const int *heads,device const int *next,device const uint *active);
float4 sampleFilm(float2 xz,device const uint4 *film,device const uint4 *dry);
float3 wetNormal(float3 p,float3 n,device const uint4 *film,device const float2 *ripples);
float3 stainColor(float3 base,float4 sample);
