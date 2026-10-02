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
constant float CREAM_DENSITY = .025; // (rho_cream - rho_coffee)/rho_coffee
constant float VORTICITY = .6;
constant float WALL_FRICTION = 2.;  // 1/s
constant float DRIFT_RATE = 9.;     // 1/s


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
constant int HALF_CELLS=CELLS/2;
// Open area fraction of cell c's lower face on axis a, from the analytic SDF at
// the face centre (0 = closed). Computed once per substep in fluidP2G; the
// solver currently treats any open fraction as fully open.
float faceFraction(int3 c,int a,device const int *type,CupBody body) {
    int3 left=c;left[a]-=1;
    if(!inGrid(left))return 1.;
    if(type[cellIndex(c)]==1 || type[cellIndex(left)]==1)return 0.;
    float d=solidDistance(ORIGIN+(float3(c)+faceOffset(a))*DX,body);
    return d<.005?0.:clamp(.5+d/DX,0.f,1.f);
}
bool blocked(int3 c,int a,device const float4 *open) {
    int3 left=c;left[a]-=1;
    if(!inGrid(c)||!inGrid(left))return false; // open simulation boundary
    return open[cellIndex(c)][a]<=0.;
}
float3 solidFace(int3 c,int a,CupBody body) {
    float3 face=ORIGIN+(float3(c)+faceOffset(a))*DX;
    return isCupContact(face,body)?cupVelocity(face,body):float3(0);
}

kernel void fluidClear(device atomic_int *heads [[buffer(1)]],device atomic_uint *active [[buffer(3)]],
                       device int *type [[buffer(6)]],device const CupBody &body [[buffer(14)]],
                       device uint *counters [[buffer(28)]],uint i [[thread_position_in_grid]]) {
    if(i>=CELLS)return;
    if(i==0)counters[0]=counters[1]=0;
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
                     device float *cream [[buffer(9)]],device const CupBody &body [[buffer(14)]],
                     device float4 *open [[buffer(25)]],device int *fluidList [[buffer(27)]],
                     device atomic_uint *counters [[buffer(28)]],uint i [[thread_position_in_grid]]) {
    if(i>=CELLS)return;
    velocity[i]=0.;mass[i]=0.;cream[i]=0.;
    if(type[i]!=1)for(int j=heads[i];j>=0;j=next[j])
        if(particles[j].cx.w<=0.){type[i]=2;break;}
    int3 c=cellCoord(i);
    if(type[i]!=2)pressure[i]=0.;
    else {
        // Red/black lists: pressure sweeps touch only liquid cells (~1%).
        int color=(c.x+c.y+c.z)&1;
        fluidList[color*HALF_CELLS+int(atomic_fetch_add_explicit(counters+color,1,memory_order_relaxed))]=int(i);
    }
    if(!active[i]){open[i]=1.;return;}
    open[i]=float4(faceFraction(c,0,type,body),faceFraction(c,1,type,body),faceFraction(c,2,type,body),0.);
    float3 center=cellCenter(c);
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
kernel void fluidIndirect(device uint *counters [[buffer(28)]],uint i [[thread_position_in_grid]]) {
    if(i)return;
    for(int color=0;color<2;color++){counters[4+4*color]=max(1u,(counters[color]+127)/128);counters[5+4*color]=1;counters[6+4*color]=1;}
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
// Cell-centred vorticity (xyz) and its magnitude (w) for confinement. Grid
// transfers smear small eddies within a few substeps; confinement restores them.
float3 centerVelocity(int3 c,device const float4 *v) {
    float3 u;int i=cellIndex(c);
    for(int a=0;a<3;a++){int3 q=c;q[a]++;u[a]=.5*(v[i][a]+(inGrid(q)?v[cellIndex(q)][a]:v[i][a]));}
    return u;
}
kernel void fluidVorticity(device const float4 *v [[buffer(4)]],device const int *type [[buffer(6)]],
                           device float4 *scratch [[buffer(26)]],uint i [[thread_position_in_grid]]) {
    if(i>=CELLS)return;
    if(type[i]!=2){scratch[i]=0.;return;}
    int3 c=cellCoord(i);float3 u0=centerVelocity(c,v),du[3];
    for(int a=0;a<3;a++) {
        int3 l=c,r=c;l[a]--;r[a]++;
        float3 ul=inGrid(l)&&type[cellIndex(l)]==2?centerVelocity(l,v):u0;
        float3 ur=inGrid(r)&&type[cellIndex(r)]==2?centerVelocity(r,v):u0;
        du[a]=(ur-ul)/(2.*DX);
    }
    float3 w=float3(du[1].z-du[2].y,du[2].x-du[0].z,du[0].y-du[1].x);
    scratch[i]=float4(w,length(w));
}
// Liquid cell with no air neighbor. Stirring and confinement act only here: on
// the free surface they roughen it into grid-scale bumps.
bool submerged(int3 c,device const int *type) {
    if(!inGrid(c)||type[cellIndex(c)]!=2)return false;
    for(int a=0;a<3;a++)for(int direction=-1;direction<=1;direction+=2) {
        int3 q=c;q[a]+=direction;if(inGrid(q)&&type[cellIndex(q)]==0)return false;
    }
    return true;
}
float3 confinement(int3 c,device const int *type,device const float4 *vort) {
    if(!submerged(c,type))return 0.;
    float4 w=vort[cellIndex(c)];float3 grad;
    for(int a=0;a<3;a++) {
        int3 l=c,r=c;l[a]--;r[a]++;
        float wl=inGrid(l)&&type[cellIndex(l)]==2?vort[cellIndex(l)].w:w.w;
        float wr=inGrid(r)&&type[cellIndex(r)]==2?vort[cellIndex(r)].w:w.w;
        grad[a]=wr-wl;
    }
    return VORTICITY*DX*cross(grad/(length(grad)+1e-5),w.xyz);
}
kernel void fluidForces(device float4 *v [[buffer(4)]],device const float4 *mass [[buffer(5)]],
                        device const float *cream [[buffer(9)]],device const uint *active [[buffer(3)]],
                        device const int *type [[buffer(6)]],device const float *s [[buffer(10)]],
                        device const CupBody &body [[buffer(14)]],device const float4 *open [[buffer(25)]],
                        device const float4 *vort [[buffer(26)]],uint i [[thread_position_in_grid]]) {
    if(i>=CELLS||!active[i])return;
    int3 c=cellCoord(i);float3 p=cellCenter(c),g=float3(s[1],s[2],s[3]);
    float3 drive=0.;
    if(s[6]>0. && inCup(p,body)) {
        // A spoon blade circling inside the cup sheds wakes and shear ribbons; a
        // weak bulk swirl keeps the liquid turning. A strong prescribed vortex
        // alone only paints concentric rings.
        float3 local=cupLocal(p,body);
        float3 relative=rotateQ(float4(-body.rotation.xyz,body.rotation.w),v[i].xyz-cupVelocity(p,body));
        relative.y=0.;
        float radius2=dot(local.xz,local.xz);
        float3 target=float3(-local.z,0,local.x)*(9.52/(1.+5.*radius2));
        float angle=s[5]*6.5;float2 spoon=.42*float2(cos(angle),sin(angle));
        float2 spoonVelocity=.42*6.5*float2(-sin(angle),cos(angle));
        float blade=submerged(c,type)?exp(-dot(local.xz-spoon,local.xz-spoon)/(.10*.10)):0.;
        float3 local3=(target-relative)*1.2+(float3(spoonVelocity.x,0,spoonVelocity.y)-relative)*blade*6.;
        drive=rotateQ(body.rotation,local3)*s[6];
    }
    if(mass[i].w>.15 && mass[i].w<7.8 && solidDistance(p,body)>.075)
        drive+=capillaryForce(c,mass);
    float3 confine=confinement(c,type,vort);
    float gravity=length(g);float3 down=g/max(gravity,1e-6f);
    float3 vel=v[i].xyz;
    for(int a=0;a<3;a++) {
        if(blocked(c,a,open)){vel[a]=solidFace(c,a,body)[a];continue;}
        int3 q=c;q[a]--;
        int lt=inGrid(q)?type[cellIndex(q)]:0,rt=type[i];
        // Body forces only on faces the projection will see; the rest are
        // overwritten by extrapolation after projection.
        if(lt!=2&&rt!=2)continue;
        // Boussinesq: cold cream is ~2.5% denser than hot coffee, so it sinks
        // as a plume and billows instead of floating. Cream averaged to the face.
        float creamFace=.5*(cream[i]+(inGrid(q)?cream[cellIndex(q)]:cream[i]));
        float confineFace=.5*(confine[a]+confinement(q,type,vort)[a]);
        // Skin friction: the grid cannot resolve the no-slip boundary layer, so
        // faces beside a wall relax toward the wall's tangential velocity. A
        // stirred cup spins down over seconds rather than (inviscidly) never.
        bool wall=false;
        for(int b=0;b<3;b++)if(b!=a)for(int direction=-1;direction<=1;direction+=2) {
            int3 r=c;r[b]+=direction;wall=wall||(inGrid(r)&&blocked(r,a,open));
        }
        // Applied before body forces: damping part of the hydrostatic g*dt
        // would leave a non-gradient remainder that the projection turns into
        // a spurious circulation in a cup at rest.
        if(wall)vel[a]=mix(vel[a],solidFace(c,a,body)[a],1.-exp(-WALL_FRICTION*s[0]));
        vel[a]+=(g[a]+down[a]*gravity*creamFace*CREAM_DENSITY+drive[a]+confineFace)*s[0];
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
                            device float *divergence [[buffer(7)]],uint i [[thread_position_in_grid]]) {
    if(i>=CELLS)return;
    // Particle/grid methods accumulate compression from wall projection and
    // source emission. Remove density drift rather than preserving particle
    // COUNT while the occupied liquid volume quietly collapses. A rate (1/s),
    // so it is independent of the adaptive substep.
    float excess=max(0.f,mass[i].w/8.-1.);
    // Superlinear in the excess: secondary flow converging on a stirred vortex's
    // axis (the tea-leaf effect) piles particles faster than a linear rate fixes.
    float drift=min(60.f,excess*DRIFT_RATE*(1.+6.*excess));
    divergence[i]=type[i]==2?divergenceAt(cellCoord(i),v)-drift:0.;
}
// Red/black SOR over compact per-colour liquid lists (indirect dispatch):
// opposite parity is read-only during each dispatch; the host inserts a GPU
// buffer barrier between colours, so there are no neighbor races.
kernel void fluidPressure(device const int *type [[buffer(6)]],device const float *divergence [[buffer(7)]],
                          device float *pressure [[buffer(8)]],device const float *s [[buffer(10)]],
                          device const float4 *open [[buffer(25)]],device const int *fluidList [[buffer(27)]],
                          device const uint *counters [[buffer(28)]],uint t [[thread_position_in_grid]]) {
    int color=int(s[12]);
    if(t>=counters[color])return;
    int i=fluidList[color*HALF_CELLS+int(t)];int3 c=cellCoord(i);
    float sum=0.,diagonal=0.;
    for(int a=0;a<3;a++)for(int direction=-1;direction<=1;direction+=2) {
        int3 q=c;q[a]+=direction;
        int3 face=direction>0?q:c;
        if(blocked(face,a,open))continue;
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
                         device const uint *active [[buffer(3)]],device const float4 *open [[buffer(25)]],
                         device const CupBody &body [[buffer(14)]],uint i [[thread_position_in_grid]]) {
    if(i>=CELLS||!active[i])return;int3 c=cellCoord(i);float3 vel=v[i].xyz;
    for(int a=0;a<3;a++) {
        int3 q=c;q[a]--;
        if(blocked(c,a,open)){vel[a]=solidFace(c,a,body)[a];continue;}
        int lt=inGrid(q)?type[cellIndex(q)]:0, rt=type[i];
        if(lt!=2&&rt!=2)continue;
        float pl=lt==2?pressure[cellIndex(q)]:0.,pr=rt==2?pressure[i]:0.;
        vel[a]-=(pr-pl)*s[0]/DX;
    }
    v[i]=float4(vel,0.);
}
// Faces with no liquid on either side are never projected, yet G2P's 3x3x3
// stencil reads them for every surface particle. Extend the projected field
// outward (two Jacobi layers, ping-pong through scratch) so surface particles
// see divergence-free velocities instead of raw P2G momentum.
bool faceKnown(int3 c,int a,device const int *type,device const float4 *open) {
    if(blocked(c,a,open))return true;
    int3 q=c;q[a]--;
    return type[cellIndex(c)]==2 || (inGrid(q)&&type[cellIndex(q)]==2);
}
kernel void fluidExtrapolate(device float4 *v [[buffer(4)]],device float4 *scratch [[buffer(26)]],
                             device const int *type [[buffer(6)]],device const uint *active [[buffer(3)]],
                             device const float4 *open [[buffer(25)]],device const float *s [[buffer(10)]],
                             uint i [[thread_position_in_grid]]) {
    if(i>=CELLS)return;
    bool first=s[22]==0.;
    device float4 *src=first?v:scratch,*dst=first?scratch:v;
    if(!active[i]){dst[i]=0.;return;}
    int3 c=cellCoord(i);float4 out=src[i];uint known=0;
    for(int a=0;a<3;a++) {
        if(first?faceKnown(c,a,type,open):((uint(src[i].w)>>a)&1u)){known|=1u<<a;continue;}
        float sum=0.,count=0.;
        for(int b=0;b<3;b++)for(int direction=-1;direction<=1;direction+=2) {
            int3 q=c;q[b]+=direction;if(!inGrid(q))continue;
            if(first?faceKnown(q,a,type,open):((uint(src[cellIndex(q)].w)>>a)&1u)){sum+=src[cellIndex(q)][a];count+=1.;}
        }
        if(count>0.){out[a]=sum/count;known|=1u<<a;}
    }
    dst[i]=float4(out.xyz,float(known));
}
float3 gridVelocity(float3 x,device const float4 *v) {
    float3 gx=(x-ORIGIN)/DX,vel=0.;
    for(int a=0;a<3;a++) {
        float3 face=gx-faceOffset(a);int3 base=int3(floor(face-.5));float total=0.;
        for(int z=0;z<3;z++)for(int y=0;y<3;y++)for(int x=0;x<3;x++) {
            int3 q=base+int3(x,y,z);if(!inGrid(q))continue;
            float w=weight(float3(q)-face);vel[a]+=w*v[cellIndex(q)][a];total+=w;
        }
        if(total>1e-6)vel[a]/=total;
    }
    return vel;
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
    float speed=length(vel);if(speed>s[23])vel*=s[23]/speed; // CFL: <= 1 cell per substep
    // Midpoint (RK2) advection: forward Euler spirals particles outward in a
    // swirl by ~r(w dt)^2/2 per step and piles them against the wall. Only the
    // displacement uses the correction; the carried velocity stays APIC.
    float3 drift=0.;
    if(!ballistic) {
        drift=gridVelocity(p.x.xyz+vel*(.5*s[0]),v)-vel;
        float m=length(drift),limit=.5*s[23];if(m>limit)drift*=limit/m;
    }
    int steps=max(1,int(ceil(length(vel+drift)*s[0]/.012)));
    bool touchedFixed=false;float fixedImpact=0.;
    for(int step=0;step<steps;step++) {
        p.x.xyz+=(vel+drift)*(s[0]/steps);
        bool floorContact=false;
        for(int k=0;k<8;k++) {
            float d=solidDistance(p.x.xyz,body);if(d>=contact)break;
            float3 n=solidNormal(p.x.xyz,body);
            p.x.xyz+=n*(contact-d+.0001);
            bool cup=isCupContact(p.x.xyz,body);
            float3 wall=cup?cupVelocity(p.x.xyz,body):float3(0);
            float vn=dot(vel-wall,n);
            float3 impulse=-n*min(0.f,vn);
            vel+=impulse;drift-=n*min(0.f,dot(drift,n));
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
            float w=1.-d*d/(radius*radius);w=w>0.?w*w*w*particleWeight(particle):0.;
            totalRadius+=particleRadius(particle)*w;
            center+=delta*w;diagonal+=delta*delta*w;
            off+=float3(delta.x*delta.y,delta.x*delta.z,delta.y*delta.z)*w;
            float mw=1.-d*d/(.06*.06);mw=mw>0.?mw*mw*mw*particleWeight(particle):0.;
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
                          device const float4 *open [[buffer(25)]],
                          device atomic_int *reaction [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=CELLS || type[i]!=2 || body.position.w<.5)return;
    int3 c=cellCoord(i);
    float density=.25/(s[18]*.0425*.0425*.0425);
    for(int a=0;a<3;a++)for(int direction=-1;direction<=1;direction+=2) {
        int3 face=c;if(direction>0)face[a]++;
        if(!blocked(face,a,open))continue;
        float3 p=ORIGIN+(float3(face)+faceOffset(a))*DX;
        if(!isCupContact(p,body))continue;
        float3 impulse=0.;
        impulse[a]=float(direction)*clamp(pressure[i],0.f,100.f)*DX*DX*s[0]*density;
        accumulateReaction(reaction,impulse,p-body.position.xyz);
    }
}
