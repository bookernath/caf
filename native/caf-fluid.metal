#if !defined(CAF_SCENE_LAYOUT) || CAF_SCENE_LAYOUT != 5
#error Rebuild caf-metal alongside the version-5 shaders
#endif
// Persistent 3D APIC liquid. Quadratic B-spline transfers on a staggered MAC
// grid, free-surface pressure projection, analytic collision constraints.
// Model units: cup height ~= 1 = 0.1 metres. Particle mass is constant; removed/off-scene
// particles keep a tombstone so diagnostics account for every emitted particle.
constant int3 GRID = int3(56,32,56);
constant float DX = .085;
constant float3 ORIGIN = float3(-2.38,-.255,-2.38);
constant int CELLS = 56*32*56;
constant int3 SURFACE = int3(224,128,224);
constant float SURFACE_DX = DX/4.;
constant float CONTACT = .013;


bool inGrid(int3 c) { return all(c>=0) && all(c<GRID); }
int cellIndex(int3 c) { return (c.z*GRID.y+c.y)*GRID.x+c.x; }
int3 cellCoord(int i) { return int3(i%GRID.x,(i/GRID.x)%GRID.y,i/(GRID.x*GRID.y)); }
float3 cellCenter(int3 c) { return ORIGIN+(float3(c)+.5)*DX; }
int3 particleCell(float3 x) { return int3(floor((x-ORIGIN)/DX)); }
float bspline(float v) {
    v=abs(v);
    return v<.5 ? .75-v*v : (v<1.5 ? .5*(1.5-v)*(1.5-v) : 0.);
}
float weight(float3 d) { return bspline(d.x)*bspline(d.y)*bspline(d.z); }
float3 faceOffset(int a) { float3 v=float3(.5); v[a]=0.; return v; }
bool blocked(int3 c,int a,device const int *type,CupBody body) {
    int3 left=c;left[a]-=1;
    if(!inGrid(c)||!inGrid(left))return false; // open simulation boundary
    return type[cellIndex(c)]==1 || type[cellIndex(left)]==1 ||
           solidDistance(ORIGIN+(float3(c)+faceOffset(a))*DX,body)<.005;
}

kernel void fluidClear(device atomic_int *heads [[buffer(1)]],device atomic_uint *active [[buffer(3)]],
                       device int *type [[buffer(6)]],device const CupBody &body [[buffer(14)]],uint i [[thread_position_in_grid]]) {
    if(i>=CELLS)return;
    atomic_store_explicit(heads+i,-1,memory_order_relaxed);
    atomic_store_explicit(active+i,0,memory_order_relaxed);
    type[i]=solidDistance(cellCenter(cellCoord(i)),body)<0. ? 1:0;
}
kernel void fluidBins(device FluidParticle *particles [[buffer(0)]],device atomic_int *heads [[buffer(1)]],
                      device int *next [[buffer(2)]],device atomic_uint *active [[buffer(3)]],
                      device const float *s [[buffer(10)]],device const CupBody &body [[buffer(14)]],device atomic_uint *details [[buffer(16)]],uint i [[thread_position_in_grid]]) {
    if(i>=atomic_load_explicit(details,memory_order_relaxed)||!particleAlive(particles[i]))return;
    int3 c=particleCell(particles[i].x.xyz);
    if(!inGrid(c))return;
    next[i]=atomic_exchange_explicit(heads+cellIndex(c),int(i),memory_order_relaxed);
    // Quadratic staggered stencils can reach two nodes along one axis.
    for(int z=-1;z<=2;z++)for(int y=-1;y<=2;y++)for(int x=-1;x<=2;x++) {
        int3 q=c+int3(x,y,z);if(inGrid(q))atomic_fetch_or_explicit(active+cellIndex(q),particles[i].cx.w>0.?3:1,memory_order_relaxed);
    }
}
kernel void fluidP2G(device const FluidParticle *particles [[buffer(0)]],device const int *heads [[buffer(1)]],
                     device const int *next [[buffer(2)]],device const uint *active [[buffer(3)]],
                     device float4 *velocity [[buffer(4)]],device float4 *mass [[buffer(5)]],
                     device int *type [[buffer(6)]],device float *pressure [[buffer(8)]],
                     device float *cream [[buffer(9)]],
                     device const CupBody &body [[buffer(14)]],uint i [[thread_position_in_grid]]) {
    if(i>=CELLS)return;
    velocity[i]=0.;mass[i]=0.;cream[i]=0.;
    if(type[i]!=1)for(int j=heads[i];j>=0;j=next[j])
        if(particles[j].cx.w<=0.){type[i]=2;break;}
    if(type[i]!=2)pressure[i]=0.;
    if(!active[i])return;
    int3 c=cellCoord(i);float3 center=cellCenter(c);
    float3 momentum=0.,weights=0.;float density=0.,creamMass=0.;
    for(int z=-2;z<=1;z++)for(int y=-2;y<=1;y++)for(int x=-2;x<=1;x++) {
        int3 bucket=c+int3(x,y,z);if(!inGrid(bucket))continue;
        for(int j=heads[cellIndex(bucket)];j>=0;j=next[j]) {
            FluidParticle p=particles[j];if(p.cx.w>0.)continue;
            float volume=particleWeight(p);
            float3 delta=(center-p.x.xyz)/DX;
            float cw=weight(delta)*volume;density+=cw;creamMass+=cw*p.v.w;
            for(int a=0;a<3;a++) {
                float3 d=delta;d[a]-=.5;
                float w=weight(d)*volume;if(w==0.)continue;
                float3 affine=a==0?p.cx.xyz:(a==1?p.cy.xyz:p.cz.xyz);
                momentum[a]+=w*(p.v[a]+dot(affine,d*DX));
                weights[a]+=w;
            }
        }
    }
    velocity[i]=float4(momentum/max(weights,float3(1e-8)),0.);
    mass[i]=float4(weights,density);cream[i]=creamMass/max(density,1e-8f);
}
float gridDensity(int3 c,device const float4 *mass) {
    return inGrid(c)?mass[cellIndex(c)].w:0.;
}
float3 surfaceGridNormal(int3 c,device const float4 *mass) {
    float3 gradient;
    for(int a=0;a<3;a++){int3 d=0;d[a]=1;gradient[a]=gridDensity(c+d,mass)-gridDensity(c-d,mass);}
    return -gradient/max(length(gradient),.05f);
}
float3 capillaryForce(int3 c,device const float4 *mass) {
    float3 n=surfaceGridNormal(c,mass);float curvature=0.;
    for(int a=0;a<3;a++) {
        int3 d=0;d[a]=1;
        curvature+=(surfaceGridNormal(c+d,mass)[a]-surfaceGridNormal(c-d,mass)[a])/(2.*DX);
    }
    // Bounded CSF force: enough cohesion for beads, not bulk viscosity or a
    // spring pulling the surface back to its original shape.
    return -n*clamp(curvature,-20.f,20.f)*.072;
}
kernel void fluidForces(device float4 *v [[buffer(4)]],device const float4 *mass [[buffer(5)]],
                        device const float *cream [[buffer(9)]],
                        device const int *type [[buffer(6)]],device const float *s [[buffer(10)]],
                        device const CupBody &body [[buffer(14)]],uint i [[thread_position_in_grid]]) {
    if(i>=CELLS)return;
    int3 c=cellCoord(i);float3 p=cellCenter(c),g=float3(s[1],s[2],s[3]);
    // Boussinesq approximation: lighter cream rises through the coffee while
    // sharing its incompressible velocity field (not a second phase solver).
    g-=g/max(length(g),1e-6f)*cream[i]*6.87;
    if(s[6]>0. && inCup(p,body)) {
        float3 local=cupLocal(p,body);
        float3 relative=rotateQ(float4(-body.rotation.xyz,body.rotation.w),v[i].xyz-cupVelocity(p,body));
        float radius2=dot(local.xz,local.xz);
        float3 target=float3(-local.z,0,local.x)*(9.52/(1.+5.*radius2));
        g+=rotateQ(body.rotation,target-float3(relative.x,0,relative.z))*s[6]*4.;
    }
    if(mass[i].w>.15 && mass[i].w<7.8 && solidDistance(p,body)>.075)
        g+=capillaryForce(c,mass);
    float3 vel=v[i].xyz;
    for(int a=0;a<3;a++) {
        if(mass[i][a]>1e-7)vel[a]+=g[a]*s[0];
        if(blocked(c,a,type,body)) {
            float3 face=ORIGIN+(float3(c)+faceOffset(a))*DX;
            vel[a]=isCupContact(face,body)?cupVelocity(face,body)[a]:0.;
        }
    }
    v[i]=float4(vel,0.);
}
float divergenceAt(int3 c,device const float4 *v) {
    float sum=0.;int i=cellIndex(c);
    for(int a=0;a<3;a++) {int3 q=c;q[a]++;sum+=(inGrid(q)?v[cellIndex(q)][a]:v[i][a])-v[i][a];}
    return sum/DX;
}
kernel void fluidDivergence(device const float4 *v [[buffer(4)]],device const int *type [[buffer(6)]],
                            device const float4 *mass [[buffer(5)]],device const float *s [[buffer(10)]],
                            device float *divergence [[buffer(7)]],device const CupBody &body [[buffer(14)]],uint i [[thread_position_in_grid]]) {
    if(i>=CELLS)return;
    // Particle/grid methods accumulate compression from wall projection and
    // source emission. Remove density drift rather than preserving particle
    // COUNT while the occupied liquid volume quietly collapses.
    float drift=min(20.f,max(0.f,mass[i].w/8.-1.)*.025/max(s[0],1e-6f));
    divergence[i]=type[i]==2?divergenceAt(cellCoord(i),v)-drift:0.;
}
// Red/black SOR: opposite parity is read-only during each dispatch. The host
// inserts a GPU buffer barrier between colors; there are no neighbor races.
kernel void fluidPressure(device const int *type [[buffer(6)]],device const float *divergence [[buffer(7)]],
                          device float *pressure [[buffer(8)]],device const float *s [[buffer(10)]],
                          device const CupBody &body [[buffer(14)]],uint i [[thread_position_in_grid]]) {
    if(i>=CELLS || type[i]!=2)return;
    int3 c=cellCoord(i);if(((c.x+c.y+c.z)&1)!=int(s[12]))return;
    float sum=0.,diagonal=0.;
    for(int a=0;a<3;a++)for(int direction=-1;direction<=1;direction+=2) {
        int3 q=c;q[a]+=direction;
        int3 face=direction>0?q:c;
        if(blocked(face,a,type,body))continue;
        diagonal+=1.;
        if(inGrid(q)&&type[cellIndex(q)]==2)sum+=pressure[cellIndex(q)]; // air is p=0
    }
    if(diagonal>0.) {
        float target=(sum-divergence[i]*DX*DX/s[0])/diagonal;
        pressure[i]=mix(pressure[i],target,s[13]);
    } else pressure[i]=0.;
}
kernel void fluidProject(device float4 *v [[buffer(4)]],device const int *type [[buffer(6)]],
                         device const float *pressure [[buffer(8)]],device const float *s [[buffer(10)]],
                         device const CupBody &body [[buffer(14)]],uint i [[thread_position_in_grid]]) {
    if(i>=CELLS)return;int3 c=cellCoord(i);float3 vel=v[i].xyz;
    for(int a=0;a<3;a++) {
        int3 q=c;q[a]--;
        if(blocked(c,a,type,body)) {
            float3 face=ORIGIN+(float3(c)+faceOffset(a))*DX;
            vel[a]=isCupContact(face,body)?cupVelocity(face,body)[a]:0.;continue;
        }
        int lt=inGrid(q)?type[cellIndex(q)]:0, rt=type[i];
        if(lt!=2&&rt!=2)continue;
        float pl=lt==2?pressure[cellIndex(q)]:0.,pr=rt==2?pressure[i]:0.;
        vel[a]-=(pr-pl)*s[0]/DX;
    }
    v[i]=float4(vel,0.);
}
kernel void fluidG2P(device FluidParticle *particles [[buffer(0)]],device const float4 *v [[buffer(4)]],
                     device atomic_int *reaction [[buffer(15)]],device atomic_uint *stats [[buffer(12)]],
                     device const float *cream [[buffer(9)]],device const float4 *mass [[buffer(5)]],
                     device atomic_uint *details [[buffer(16)]],device atomic_uint *pending [[buffer(17)]],
                     device const uint4 *film [[buffer(18)]],device float4 *ripples [[buffer(24)]],
                     device const float *s [[buffer(10)]],device const CupBody &body [[buffer(14)]],uint i [[thread_position_in_grid]]) {
    if(i>=uint(s[4]))return;FluidParticle p=particles[i];if(!particleAlive(p))return;
    float3 gx=(p.x.xyz-ORIGIN)/DX,vel=0.;float3 affine[3]={float3(0),float3(0),float3(0)};
    bool ballistic=p.cx.w>0.;
    for(int a=0;a<3 && !ballistic;a++) {
        float3 face=gx-faceOffset(a);int3 base=int3(floor(face-.5));
        float total=0.;
        for(int z=0;z<3;z++)for(int y=0;y<3;y++)for(int x=0;x<3;x++) {
            int3 q=base+int3(x,y,z);if(!inGrid(q))continue;
            float3 delta=float3(q)-face;float w=weight(delta),gv=v[cellIndex(q)][a];
            vel[a]+=w*gv;affine[a]+=w*gv*(delta*DX);total+=w;
        }
        if(total>1e-6){vel[a]/=total;affine[a]*=4./(DX*DX*total);}
    }
    if(ballistic) {
        vel=(p.v.xyz+float3(s[1],s[2],s[3])*s[0])*exp(-.06*s[0]);
        p.cx.w+=s[0];
        int3 cell=particleCell(p.x.xyz);
        if(inGrid(cell)&&mass[cellIndex(cell)].w>2.5 && p.cx.w>.025) {
            p.cx.w=0.;
            if(inCup(p.x.xyz,body) && vel.y<-1.) {
                // Bound writes per frame: modulo alone could race when more
                // than 64 drops land in the same parallel dispatch.
                uint accepted=atomic_fetch_add_explicit(details+11,1,memory_order_relaxed);
                if(accepted<8) {
                    uint slot=atomic_fetch_add_explicit(details+7,1,memory_order_relaxed)%64;
                    float3 local=cupLocal(p.x.xyz,body);
                    ripples[slot]=float4(local.xz,s[5],min(.003f,length(vel)*particleWeight(p)*.0005));
                }
            }
        }
    }
    float contact=ballistic?particleRadius(p):CONTACT*pow(max(.125f,particleWeight(p)),1./3.);
    // A short-range capillary attraction lets the coffee follow the rolled
    // lip briefly before detaching. This is local rim adhesion, not a spring
    // tethering escaped drops back to the cup or persistent surface stains.
    float3 local=cupLocal(p.x.xyz,body);
    float rimDistance=cupShape(local);
    if(local.y>.9 && local.y<1.10 && length(local.xz)>.72 && rimDistance>contact && rimDistance<.06)
        vel-=solidNormal(p.x.xyz,body)*(2.8*(1.-smoothstep(contact,.06f,rimDistance))*s[0]);
    // Collision substeps are smaller than the wall thickness. Position
    // projection and normal-velocity removal stop thin-rim tunnelling.
    float speed=length(vel);if(speed>18.)vel*=18./speed;
    int steps=max(1,int(ceil(length(vel)*s[0]/.012)));
    bool touchedFixed=false;float fixedImpact=0.;
    for(int step=0;step<steps;step++) {
        p.x.xyz+=vel*(s[0]/steps);
        bool floorContact=false;
        for(int k=0;k<8;k++) {
            float d=solidDistance(p.x.xyz,body);if(d>=contact)break;
            float3 n=solidNormal(p.x.xyz,body);
            p.x.xyz+=n*(contact-d+.0001);
            bool cup=isCupContact(p.x.xyz,body);
            float3 wall=cup?cupVelocity(p.x.xyz,body):float3(0);
            float vn=dot(vel-wall,n);
            float3 impulse=-n*min(0.f,vn);
            vel+=impulse;
            if(cup && body.position.w>.5)accumulateReaction(reaction,-impulse*(.25/s[18])*particleWeight(p),p.x.xyz-body.position.xyz);
            if(vn<-.7)atomic_fetch_add_explicit(stats+22,uint(min(30.f,-vn-.7)*particleWeight(p)*100.),memory_order_relaxed);
            if(!cup){floorContact=touchedFixed=true;fixedImpact=max(fixedImpact,-vn);}
        }
        // Apply tangential floor drag once per elapsed collision substep, not
        // once per projection iteration (which depended on contact geometry).
        if(floorContact) {
            float3 n=fixedNormal(p.x.xyz);
            float3 normalVelocity=n*dot(vel,n);
            vel=normalVelocity+(vel-normalVelocity)*exp(-1.2*s[0]/steps);
        }
    }
    if(touchedFixed)depositFilm(p,fixedImpact,film,pending,details);
    if(!inGrid(particleCell(p.x.xyz)))p.cz.w=1.; // off-scene, accounted, never respawned
    // Advected material concentration is carried by equal-mass particles.
    // A small PIC scalar blend models diffusion without changing fluid forces.
    float milk=0.,totalMilk=0.;int3 dyeBase=int3(floor(gx-1.));
    for(int z=0;z<3;z++)for(int y=0;y<3;y++)for(int x=0;x<3;x++) {
        int3 c=dyeBase+int3(x,y,z);if(!inGrid(c))continue;
        float w=weight(float3(c)+.5-gx);milk+=w*cream[cellIndex(c)];totalMilk+=w;
    }
    float concentration=!ballistic && totalMilk>1e-6?mix(p.v.w,milk/totalMilk,1.-exp(-s[0]*.12)):p.v.w;
    p.v=float4(vel,clamp(concentration,0.f,1.f));
    p.cx=float4(clamp(affine[0],-100.f,100.f),p.cx.w);
    p.cy=float4(clamp(affine[1],-100.f,100.f),p.cy.w);
    p.cz=float4(clamp(affine[2],-100.f,100.f),p.cz.w);
    particles[i]=p;
}
kernel void fluidEvents(device FluidParticle *particles [[buffer(0)]],device const float *s [[buffer(10)]],
                        device atomic_uint *stats [[buffer(12)]],device const CupBody &body [[buffer(14)]],uint i [[thread_position_in_grid]]) {
    if(i>=uint(s[4]))return;
    FluidParticle p=particles[i];
    if(i>=uint(s[10])&&i<uint(s[11])) {
        bool milk=i>=uint(s[16]) && i<uint(s[17]);
        float a=hash21(float2(i,7))*6.2831853,r=sqrt(hash21(float2(i,11)))*(milk?.044:.108);
        float fall=milk?5.75:6.64;
        float2 offset=milk ? .26*float2(cos(s[5]*2.),sin(s[5]*2.)) : float2(-.15,-.18);
        float3 nozzle=body.position.xyz+float3(offset.x,.93,offset.y);
        p.x=float4(nozzle+float3(cos(a)*r,hash21(float2(i,13))*fall*s[20],sin(a)*r),8.);
        p.v=float4(0,-fall,0,milk?1.:0.);p.cx=p.cy=p.cz=0.;
    }
    if(particleAlive(p) && inCup(p.x.xyz,body)) {
        if(s[8]>0 && atomic_fetch_add_explicit(stats+10,uint(p.x.w),memory_order_relaxed)<uint(s[8])*8)p.cz.w=2.;
        p.v.xyz+=float3(cos(s[7]),.1,-sin(s[7]))*s[9];
    }
    particles[i]=p;
}
// Fine, covariance-aware moving-least-squares reconstruction. A sparse
// occupancy mask skips empty space; local normal-direction variance thins
// sheets/puddles without forcing the liquid back into the cup or deleting mass.
kernel void fluidSurface(device const FluidParticle *particles [[buffer(0)]],device const int *heads [[buffer(1)]],
                         device const int *next [[buffer(2)]],device const uint *active [[buffer(3)]],
                         device float4 *field [[buffer(11)]],uint i [[thread_position_in_grid]]) {
    if(i>=uint(SURFACE.x*SURFACE.y*SURFACE.z))return;
    int3 fine=int3(i%SURFACE.x,(i/SURFACE.x)%SURFACE.y,i/(SURFACE.x*SURFACE.y));
    float3 p=ORIGIN+(float3(fine)+.5)*SURFACE_DX;int3 c=particleCell(p);
    if(!inGrid(c)||!active[cellIndex(c)]){field[i]=float4(.12,0,0,0);return;}
    float3 center=0.,diagonal=0.,off=0.;float total=0.,milk=0.,milkWeight=0.,coarseMilk=0.,nearest=.12,totalRadius=0.;
    float radius=DX*1.25;
    for(int z=-1;z<=1;z++)for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++) {
        int3 q=c+int3(x,y,z);if(!inGrid(q))continue;
        for(int j=heads[cellIndex(q)];j>=0;j=next[j]) {
            FluidParticle particle=particles[j];if(particle.cx.w>0.)continue;float3 delta=particle.x.xyz-p;
            float d=length(delta);nearest=min(nearest,d-particleRadius(particle));
            float w=pow(max(0.f,1.-d*d/(radius*radius)),3.f)*particleWeight(particle);
            totalRadius+=particleRadius(particle)*w;
            center+=delta*w;diagonal+=delta*delta*w;
            off+=float3(delta.x*delta.y,delta.x*delta.z,delta.y*delta.z)*w;
            float mw=pow(max(0.f,1.-d*d/(.06*.06)),3.f)*particleWeight(particle);
            milk+=particle.v.w*mw;milkWeight+=mw;coarseMilk+=particle.v.w*w;total+=w;
        }
    }
    float phi=nearest;
    if(total>1e-6) {
        center/=total;diagonal=diagonal/total-center*center;
        off=off/total-float3(center.x*center.y,center.x*center.z,center.y*center.z);
        float distance=length(center);float3 n=center/max(distance,1e-6f);
        float variance=dot(n*n,diagonal)+2.*dot(float3(n.x*n.y,n.x*n.z,n.y*n.z),off);
        float thickness=clamp(.012+.50*sqrt(max(0.f,variance)),.014f,.034f);
        // Isolated droplets retain their own radius rather than stretching
        // toward unrelated neighbors. Dense sheets use the local covariance.
        thickness*=clamp(totalRadius/total/.026,.5f,1.f);
        thickness=mix(totalRadius/total,thickness,smoothstep(.5f,2.f,total));
        phi=distance-thickness;
    }
    float concentration=milkWeight>1e-6?milk/milkWeight:(total>1e-6?coarseMilk/total:0.);
    field[i]=float4(clamp(phi,-.1f,.12f),concentration,total,0.);
}
float4 fluidSample(float3 p,device const float *raw) {
    device const float4 *field=reinterpret_cast<device const float4 *>(raw);
    float3 high=ORIGIN+float3(GRID)*DX;
    float3 outside=max(max(ORIGIN-p,p-high),float3(0));
    if(any(outside>0.))return float4(length(outside)+.015,0,0,0);
    float3 g=clamp((p-ORIGIN)/SURFACE_DX-.5,float3(0),float3(SURFACE)-1.001);
    int3 i=int3(g);float3 f=fract(g);float4 value=0.;
    for(int z=0;z<2;z++)for(int y=0;y<2;y++)for(int x=0;x<2;x++) {
        int3 c=i+int3(x,y,z);int k=(c.z*SURFACE.y+c.y)*SURFACE.x+c.x;
        value+=field[k]*(x?f.x:1.-f.x)*(y?f.y:1.-f.y)*(z?f.z:1.-f.z);
    }
    return value;
}
float fluidDistance(float3 p,device const float *field,CupBody body) {
    return max(fluidSample(p,field).x,.001-solidDistance(p,body));
}
kernel void fluidStats(device const FluidParticle *particles [[buffer(0)]],device const float *s [[buffer(10)]],
                       device atomic_uint *stats [[buffer(12)]],device atomic_uint *details [[buffer(16)]],
                       device const CupBody &body [[buffer(14)]],uint i [[thread_position_in_grid]]) {
    if(i>=atomic_load_explicit(details,memory_order_relaxed))return;FluidParticle p=particles[i];
    if(!all(isfinite(p.x))||!all(isfinite(p.v))||!isfinite(p.cz.w)) {
        atomic_fetch_add_explicit(stats+6,8,memory_order_relaxed);return;
    }
    float volume=particleWeight(p);uint units=uint(p.x.w);
    atomic_fetch_add_explicit(stats+14,uint(clamp(p.v.w,0.f,1.f)*volume*65536.),memory_order_relaxed);
    if(p.cz.w==1.) {atomic_fetch_add_explicit(stats+4,units,memory_order_relaxed);return;}
    if(p.cz.w==2.) {atomic_fetch_add_explicit(stats+5,units,memory_order_relaxed);return;}
    if(!particleAlive(p))return;
    int category=inCup(p.x.xyz,body)?0:(p.x.y<.18 && length(p.x.xz)<1.31?1:(p.x.y<.04?2:3));
    atomic_fetch_add_explicit(stats+category,units,memory_order_relaxed);
    atomic_fetch_add_explicit(stats+7,uint(min(64000.f,dot(p.v.xyz,p.v.xyz))*volume*100.),memory_order_relaxed);
    atomic_fetch_add_explicit(stats+48,units,memory_order_relaxed);
    if(p.cx.w>0.) {
        atomic_fetch_add_explicit(stats+40,1,memory_order_relaxed);
        atomic_fetch_max_explicit(stats+43,uint(particleRadius(p)*1e6),memory_order_relaxed);
        atomic_fetch_add_explicit(stats+41,units,memory_order_relaxed);
    }
    atomic_fetch_max_explicit(stats+31,uint(min(64000.f,dot(p.v.xyz,p.v.xyz))*1000.),memory_order_relaxed);
    atomic_fetch_max_explicit(stats+8,uint(max(0.f,-solidDistance(p.x.xyz,body))*1000000.),memory_order_relaxed);
    if(category==0) {
        atomic_fetch_max_explicit(stats+9,uint(max(0.f,p.x.y)*1000000.),memory_order_relaxed);
        atomic_fetch_add_explicit(stats+15,uint(clamp(p.v.w,0.f,1.f)*volume*65536.),memory_order_relaxed);
        atomic_fetch_add_explicit(stats+16,uint(clamp(p.v.w*p.v.w,0.f,1.f)*volume*65536.),memory_order_relaxed);
        atomic_fetch_add_explicit(stats+26,uint(max(0.f,cupLocal(p.x.xyz,body).y)*volume*1000.),memory_order_relaxed);
    }
}
kernel void fluidGridStats(device const float4 *v [[buffer(4)]],device const int *type [[buffer(6)]],
                           device const float4 *mass [[buffer(5)]],
                           device const float *divergence [[buffer(7)]],device atomic_uint *stats [[buffer(12)]],
                           device const CupBody &body [[buffer(14)]],uint i [[thread_position_in_grid]]) {
    if(i>=CELLS||type[i]!=2)return;
    atomic_fetch_add_explicit(stats+11,uint(min(100000.f,abs(divergence[i])*1000.)),memory_order_relaxed);
    atomic_fetch_add_explicit(stats+12,uint(min(100000.f,abs(divergenceAt(cellCoord(i),v))*1000.)),memory_order_relaxed);
    atomic_fetch_add_explicit(stats+13,1,memory_order_relaxed);
    atomic_fetch_max_explicit(stats+27,uint(mass[i].w*1000.),memory_order_relaxed);
    atomic_fetch_add_explicit(stats+28,1,memory_order_relaxed);
    if(mass[i].w>10.)atomic_fetch_add_explicit(stats+29,1,memory_order_relaxed);
    atomic_fetch_add_explicit(stats+30,uint(min(1.f,mass[i].w/8.)*DX*DX*DX*1e6),memory_order_relaxed);
}

// Integrate fluid pressure traction on the moving shell. Particle collision
// impulses account for contacts below grid resolution; this handles bulk load.
kernel void fluidReaction(device const int *type [[buffer(6)]],device const float *pressure [[buffer(8)]],
                          device const float *s [[buffer(10)]],device const CupBody &body [[buffer(14)]],
                          device atomic_int *reaction [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=CELLS || type[i]!=2 || body.position.w<.5)return;
    int3 c=cellCoord(i);
    float density=.25/(s[18]*.0425*.0425*.0425);
    for(int a=0;a<3;a++)for(int direction=-1;direction<=1;direction+=2) {
        int3 face=c;if(direction>0)face[a]++;
        if(!blocked(face,a,type,body))continue;
        float3 p=ORIGIN+(float3(face)+faceOffset(a))*DX;
        if(!isCupContact(p,body))continue;
        float3 impulse=0.;
        impulse[a]=float(direction)*clamp(pressure[i],0.f,100.f)*DX*DX*s[0]*density;
        accumulateReaction(reaction,impulse,p-body.position.xyz);
    }
}
