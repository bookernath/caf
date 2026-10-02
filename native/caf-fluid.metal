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
// Multigrid levels for the pressure preconditioner (56->28->14).
constant int3 GRID1=int3(28,16,28),GRID2=int3(14,8,14);
constant int CELLS1=28*16*28,CELLS2=14*8*14;
constant int SOLVE_ITERATIONS=40;
constant float SOLVE_TOLERANCE=.0005; // RMS divergence residual, 1/s
constant int COARSE_SWEEPS=8;
constant float THETA_MIN=.25;
constant float COARSE_GAIN=1.;
// Per coarse cell: lower-face weights + Dirichlet diagonal, unknown, rhs,
// residual; `list` holds the level's red (from 0) / black (from end) lists.
struct MGCell { float4 w; float x,b,r; int list; };
// Solver buffer (29): float4 coefficients[CELLS], then float phi, r, z, d, q.
device float *solverVector(device float4 *solver,int k) { return reinterpret_cast<device float *>(solver+CELLS)+k*CELLS; }
// Open area fraction of cell c's lower face on axis a (0 = closed), from a
// planar fit of the cup SDF across the DX square. Recomputed every substep in
// fluidP2G; weights the pressure matrix and divergence (variational cut cells).
float faceFraction(int3 c,int a,device const int *type,CupBody body) {
    int3 left=c;left[a]-=1;
    if(!inGrid(left))return 1.;
    // Walls are ~1.1 cells thick, so a face spanning one always has a solid
    // cell centre beside it: this keeps thin walls leak-proof.
    if(type[cellIndex(c)]==1 || type[cellIndex(left)]==1)return 0.;
    float3 x=ORIGIN+(float3(c)+faceOffset(a))*DX;
    float cup=cupShape(cupLocal(x,body)),fixed=fixedSolid(x),d=min(cup,fixed);
    // The flat table/saucer stay binary: sub-cell weights there only let
    // spilled films slip through the projection laterally and clump.
    if(fixed<=cup)return d<.005?0.:1.;
    if(d>DX)return 1.;
    float3 n=solidNormal(x,body);
    float spread=.5*DX*(abs(n[(a+1)%3])+abs(n[(a+2)%3]));
    float f=spread<1e-4?(d>0.?1.:0.):clamp(.5+.5*d/spread,0.f,1.f);
    return f<.05?0.:f;
}
// Signed distance (cells, + into air) of a cell centre from a flat surface,
// given its fill fraction: inverts the quadratic B-spline's smoothed step.
float fillDistance(float f) {
    if(f<=1./6.)return 1.5-pow(6.*max(f,0.f),1./3.);
    if(f>=5./6.)return pow(6.*max(1.-f,0.f),1./3.)-1.5;
    float d=(.5-f)/.75;
    for(int k=0;k<3;k++)d-=(.75*d-d*d*d/3.-(.5-f))/(.75-d*d);
    return d;
}
// Ghost fluid: the fraction of the liquid-air centre spacing that is liquid,
// from the coarse level set. Pressure unknowns stay every particle cell, so
// sub-cell films stay incompressible. theta >= 0.25, not 0.01: cells barely
// inside (or outside) the level set would pin p ~ 0, so a plunging pour or
// cream stream splats on the surface instead of penetrating, and the top
// layer can no longer carry a denser plume's weight.
float ghostTheta(float liquid,float air) {
    if(air<=0. || air<=liquid)return 1.;
    return clamp(liquid/(liquid-air),THETA_MIN,1.f);
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
                     device atomic_uint *counters [[buffer(28)]],device float4 *solver [[buffer(29)]],uint i [[thread_position_in_grid]]) {
    if(i>=CELLS)return;
    device float *phi=solverVector(solver,0);
    velocity[i]=0.;mass[i]=0.;cream[i]=0.;
    if(type[i]!=1)for(int j=heads[i];j>=0;j=next[j])
        if(particles[j].cx.w<=0.){type[i]=2;break;}
    int3 c=cellCoord(i);
    if(!active[i]){open[i]=1.;phi[i]=1.5*DX;pressure[i]=0.;return;}
    open[i]=float4(faceFraction(c,0,type,body),faceFraction(c,1,type,body),faceFraction(c,2,type,body),0.);
    float3 center=cellCenter(c);
    float3 momentum=0.,weights=0.;float density=0.,creamMass=0.,share=0.;
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
    // Coarse surface level set for the ghost-fluid boundary. The fill is
    // normalised by the kernel's non-solid share, so liquid against a wall
    // reads as full rather than as a free surface.
    for(int z=-1;z<=1;z++)for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++) {
        int3 q=c+int3(x,y,z);
        if(!inGrid(q)||type[cellIndex(q)]!=1)share+=weight(float3(x,y,z));
    }
    phi[i]=fillDistance(density/(8.*max(share,.25f)))*DX;
    if(type[i]!=2)pressure[i]=0.;
    else {
        // Red/black lists: the pressure solve touches only liquid cells (~1%).
        int color=(c.x+c.y+c.z)&1;
        fluidList[color*HALF_CELLS+int(atomic_fetch_add_explicit(counters+color,1,memory_order_relaxed))]=int(i);
    }
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
// Variational (Batty) divergence: each face's flux is its open share of the
// liquid velocity plus the closed share moving with the wall.
float faceFlux(int3 c,int a,device const float4 *v,device const float4 *open,CupBody body) {
    float f=open[cellIndex(c)][a],u=v[cellIndex(c)][a];
    return f>=1.?u:f*u+(1.-f)*solidFace(c,a,body)[a];
}
float cutDivergence(int3 c,device const float4 *v,device const float4 *open,CupBody body) {
    float sum=0.;int i=cellIndex(c);
    for(int a=0;a<3;a++) {
        int3 q=c;q[a]++;
        sum+=(inGrid(q)?faceFlux(q,a,v,open,body):v[i][a])-faceFlux(c,a,v,open,body);
    }
    return sum/DX;
}
bool solveCell(int i,device const int *type) { return type[i]==2; }
// Divergence target and matrix coefficients: lower-face weights (open area,
// liquid on both sides) and the Dirichlet diagonal open/theta toward air.
kernel void fluidDivergence(device const float4 *v [[buffer(4)]],device const int *type [[buffer(6)]],
                            device const float4 *mass [[buffer(5)]],device const float *s [[buffer(10)]],
                            device float *divergence [[buffer(7)]],device const float4 *open [[buffer(25)]],
                            device float4 *solver [[buffer(29)]],device const CupBody &body [[buffer(14)]],
                            uint i [[thread_position_in_grid]]) {
    if(i>=CELLS)return;
    device const float *phi=solverVector(solver,0);
    if(!solveCell(i,type)){solver[i]=0.;divergence[i]=0.;return;}
    int3 c=cellCoord(i);float4 w=float4(0,0,0,1e-4); // tiny shift: enclosed pockets stay SPD
    for(int a=0;a<3;a++)for(int direction=-1;direction<=1;direction+=2) {
        int3 q=c;q[a]+=direction;int3 face=direction>0?q:c;
        float f=inGrid(face)?open[cellIndex(face)][a]:1.;
        if(f<=0.)continue;
        if(inGrid(q)&&solveCell(cellIndex(q),type)){if(direction<0)w[a]=f;}
        else w.w+=f/(inGrid(q)?ghostTheta(phi[i],phi[cellIndex(q)]):1.);
    }
    solver[i]=w;
    // Particle/grid methods accumulate compression from wall projection and
    // source emission. Remove density drift rather than preserving particle
    // COUNT while the occupied liquid volume quietly collapses. A rate (1/s),
    // so it is independent of the adaptive substep.
    float excess=max(0.f,mass[i].w/8.-1.);
    // Superlinear in the excess: secondary flow converging on a stirred vortex's
    // axis (the tea-leaf effect) piles particles faster than a linear rate fixes.
    float drift=min(60.f,excess*DRIFT_RATE*(1.+6.*excess));
    divergence[i]=cutDivergence(c,v,open,body)-drift;
}
// MGPCG pressure solve in one threadgroup: threadgroup barriers replace the
// per-sweep dispatches. Preconditioner: Galerkin V-cycle over piecewise-
// constant aggregation (coarse face weight = summed child face weights) with
// symmetric red/black Gauss-Seidel, so CG sees a fixed SPD operator.
#define SYNC threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup)
#define FOR(n) for(uint k=t;k<uint(n);k+=T)
float groupSum(float v,threadgroup float *partial,uint lane,uint warp,uint warps) {
    v=simd_sum(v);if(lane==0)partial[warp]=v;
    SYNC;
    float total=0.;for(uint k=0;k<warps;k++)total+=partial[k];
    SYNC;
    return total;
}
// Off-diagonal sum (sum w_j v_j) and diagonal of fine row i.
float fineRow(int i,device const float4 *coef,device const float *v,thread float &diagonal) {
    int3 c=cellCoord(i);float4 w=coef[i];float sum=0.;diagonal=w.w+w.x+w.y+w.z;
    const int sy=GRID.x,sz=GRID.x*GRID.y;
    if(w.x>0.)sum+=w.x*v[i-1];
    if(w.y>0.)sum+=w.y*v[i-sy];
    if(w.z>0.)sum+=w.z*v[i-sz];
    if(c.x+1<GRID.x){float u=coef[i+1].x;diagonal+=u;if(u>0.)sum+=u*v[i+1];}
    if(c.y+1<GRID.y){float u=coef[i+sy].y;diagonal+=u;if(u>0.)sum+=u*v[i+sy];}
    if(c.z+1<GRID.z){float u=coef[i+sz].z;diagonal+=u;if(u>0.)sum+=u*v[i+sz];}
    return sum;
}
float fineDiagonal(int i,device const float4 *coef) {
    int3 c=cellCoord(i);float4 w=coef[i];float diagonal=w.w+w.x+w.y+w.z;
    if(c.x+1<GRID.x)diagonal+=coef[i+1].x;
    if(c.y+1<GRID.y)diagonal+=coef[i+GRID.x].y;
    if(c.z+1<GRID.z)diagonal+=coef[i+GRID.x*GRID.y].z;
    return diagonal;
}
int3 levelCoord(int i,int3 g) { return int3(i%g.x,(i/g.x)%g.y,i/(g.x*g.y)); }
int levelIndex(int3 c,int3 g) { return (c.z*g.y+c.y)*g.x+c.x; }
float coarseRow(device const MGCell *m,int i,int3 g,thread float &diagonal) {
    int3 c=levelCoord(i,g);float4 w=m[i].w;float sum=0.;diagonal=w.w+w.x+w.y+w.z;
    int sy=g.x,sz=g.x*g.y;
    if(w.x>0.)sum+=w.x*m[i-1].x;
    if(w.y>0.)sum+=w.y*m[i-sy].x;
    if(w.z>0.)sum+=w.z*m[i-sz].x;
    if(c.x+1<g.x){float u=m[i+1].w.x;diagonal+=u;if(u>0.)sum+=u*m[i+1].x;}
    if(c.y+1<g.y){float u=m[i+sy].w.y;diagonal+=u;if(u>0.)sum+=u*m[i+sy].x;}
    if(c.z+1<g.z){float u=m[i+sz].w.z;diagonal+=u;if(u>0.)sum+=u*m[i+sz].x;}
    return sum;
}
int fineCell(device const int *list,int red,uint k) { return list[k<uint(red)?int(k):HALF_CELLS+int(k)-red]; }
int coarseCell(device const MGCell *m,int count,int red,uint k) { return k<uint(red)?m[k].list:m[count-1-(int(k)-red)].list; }
// Red/black Gauss-Seidel: colour 0 = red list, 1 = black list.
void coarseSweep(device MGCell *m,int count,int2 n,int color,uint t,uint T,int3 g) {
    FOR(color?n.y:n.x) {
        int i=coarseCell(m,count,n.x,color?k+uint(n.x):k);float diagonal;
        float sum=coarseRow(m,i,g,diagonal);m[i].x=(m[i].b+sum)/diagonal;
    }
    SYNC;
}
void fineSweep(device const int *list,int red,int black,int color,device const float4 *coef,
               device const float *rhs,device float *z,uint t,uint T) {
    FOR(color?black:red) {
        int i=list[color*HALF_CELLS+int(k)];float diagonal;
        float sum=fineRow(i,coef,z,diagonal);z[i]=(rhs[i]+sum)/diagonal;
    }
    SYNC;
}
// Coarse level from its finer level: Galerkin R A P with piecewise-constant P.
// Internal child faces cancel; external ones sum into the coarse face weight.
template<typename Fine>
void buildLevel(device MGCell *m,int count,int3 g,Fine fine,int3 fg,threadgroup atomic_int *counts,uint t,uint T) {
    FOR(count) {
        int3 c=levelCoord(int(k),g)*2;float4 w=0.;
        for(int o=0;o<8;o++) {
            float4 child=fine(levelIndex(c+int3(o&1,(o>>1)&1,o>>2),fg));
            w.w+=child.w;if(!(o&1))w.x+=child.x;if(!(o&2))w.y+=child.y;if(!(o&4))w.z+=child.z;
        }
        m[k].w=w;m[k].x=0.;
        if(w.w>0.) {
            int3 cc=levelCoord(int(k),g);int color=(cc.x+cc.y+cc.z)&1;
            int slot=atomic_fetch_add_explicit(counts+color,1,memory_order_relaxed);
            m[color?count-1-slot:slot].list=int(k);
        }
    }
}
struct FineWeights { device const float4 *coef; float4 operator()(int i) const { return coef[i]; } };
struct CoarseWeights { device const MGCell *m; float4 operator()(int i) const { return m[i].w; } };
// z = M^-1 r. Pre-smoothing red,black from zero; post-smoothing black,red.
void vcycle(device const int *list,int red,int black,device const float4 *coef,device const float *r,
            device float *z,device float *q,device MGCell *m1,int2 n1,device MGCell *m2,int2 n2,uint t,uint T) {
    FOR(red+black){int i=fineCell(list,red,k);z[i]=k<uint(red)?r[i]/fineDiagonal(i,coef):0.;}
    SYNC;
    fineSweep(list,red,black,1,coef,r,z,t,T);
    FOR(red+black){int i=fineCell(list,red,k);float diagonal;float sum=fineRow(i,coef,z,diagonal);q[i]=r[i]-(diagonal*z[i]-sum);}
    SYNC;
    FOR(n1.x+n1.y) {
        int I=coarseCell(m1,CELLS1,n1.x,k);int3 c=levelCoord(I,GRID1)*2;float b=0.;
        for(int o=0;o<8;o++){int i=cellIndex(c+int3(o&1,(o>>1)&1,o>>2));if(coef[i].w>0.)b+=q[i];}
        m1[I].b=b;m1[I].x=0.;
    }
    SYNC;
    coarseSweep(m1,CELLS1,n1,0,t,T,GRID1);
    coarseSweep(m1,CELLS1,n1,1,t,T,GRID1);
    FOR(n1.x+n1.y){int I=coarseCell(m1,CELLS1,n1.x,k);float diagonal;float sum=coarseRow(m1,I,GRID1,diagonal);m1[I].r=m1[I].b-(diagonal*m1[I].x-sum);}
    SYNC;
    FOR(n2.x+n2.y) {
        int I=coarseCell(m2,CELLS2,n2.x,k);int3 c=levelCoord(I,GRID2)*2;float b=0.;
        for(int o=0;o<8;o++){int j=levelIndex(c+int3(o&1,(o>>1)&1,o>>2),GRID1);if(m1[j].w.w>0.)b+=m1[j].r;}
        m2[I].b=b;m2[I].x=0.;
    }
    SYNC;
    // Symmetric (forward then reverse) sweeps on the 14^3 level.
    for(int sweep=0;sweep<COARSE_SWEEPS;sweep++)coarseSweep(m2,CELLS2,n2,(sweep<COARSE_SWEEPS/2?sweep:sweep+1)&1,t,T,GRID2);
    FOR(n1.x+n1.y){int I=coarseCell(m1,CELLS1,n1.x,k);m1[I].x+=COARSE_GAIN*m2[levelIndex(levelCoord(I,GRID1)/2,GRID2)].x;}
    SYNC;
    coarseSweep(m1,CELLS1,n1,1,t,T,GRID1);
    coarseSweep(m1,CELLS1,n1,0,t,T,GRID1);
    FOR(red+black){int i=fineCell(list,red,k);z[i]+=COARSE_GAIN*m1[levelIndex(cellCoord(i)/2,GRID1)].x;}
    SYNC;
    fineSweep(list,red,black,1,coef,r,z,t,T);
    fineSweep(list,red,black,0,coef,r,z,t,T);
}
kernel void fluidSolve(device const float *divergence [[buffer(7)]],
                       device float *pressure [[buffer(8)]],device const float *s [[buffer(10)]],
                       device atomic_uint *stats [[buffer(12)]],device const int *list [[buffer(27)]],
                       device const uint *counters [[buffer(28)]],device float4 *solver [[buffer(29)]],
                       device MGCell *mg [[buffer(30)]],
                       uint t [[thread_index_in_threadgroup]],uint T [[threads_per_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],uint warp [[simdgroup_index_in_threadgroup]],
                       uint warps [[simdgroups_per_threadgroup]]) {
    threadgroup float partial[32];threadgroup atomic_int counts[4];
    device const float4 *coef=solver;
    device float *r=solverVector(solver,1),*z=solverVector(solver,2),*d=solverVector(solver,3),*q=solverVector(solver,4);
    device float *x=pressure; // warm start: last substep's pressure, 0 off the liquid
    device MGCell *m1=mg,*m2=mg+CELLS1;
    int red=int(counters[0]),black=int(counters[1]);uint n=uint(red+black);
    if(t<4)atomic_store_explicit(counts+t,0,memory_order_relaxed);
    SYNC;
    buildLevel(m1,CELLS1,GRID1,FineWeights{coef},GRID,counts,t,T);
    SYNC;
    buildLevel(m2,CELLS2,GRID2,CoarseWeights{m1},GRID1,counts+2,t,T);
    float scale=DX*DX/s[0],local=0.;
    FOR(n) {
        int i=fineCell(list,red,k);float diagonal;
        float sum=fineRow(i,coef,x,diagonal);
        r[i]=-divergence[i]*scale-(diagonal*x[i]-sum);local+=r[i]*r[i];
    }
    float rr=groupSum(local,partial,lane,warp,warps);
    int2 n1=int2(atomic_load_explicit(counts,memory_order_relaxed),atomic_load_explicit(counts+1,memory_order_relaxed));
    int2 n2=int2(atomic_load_explicit(counts+2,memory_order_relaxed),atomic_load_explicit(counts+3,memory_order_relaxed));
    float limit=SOLVE_TOLERANCE*SOLVE_TOLERANCE*scale*scale*float(n);
    int iterations=0;
    if(rr>limit) {
        vcycle(list,red,black,coef,r,z,q,m1,n1,m2,n2,t,T);
        local=0.;FOR(n){int i=fineCell(list,red,k);d[i]=z[i];local+=r[i]*z[i];}
        float rz=groupSum(local,partial,lane,warp,warps);
        while(iterations<SOLVE_ITERATIONS) {
            iterations++;
            local=0.;
            FOR(n){int i=fineCell(list,red,k);float diagonal;float sum=fineRow(i,coef,d,diagonal);q[i]=diagonal*d[i]-sum;local+=d[i]*q[i];}
            float alpha=rz/max(groupSum(local,partial,lane,warp,warps),1e-30f);
            local=0.;
            FOR(n){int i=fineCell(list,red,k);x[i]+=alpha*d[i];r[i]-=alpha*q[i];local+=r[i]*r[i];}
            rr=groupSum(local,partial,lane,warp,warps);
            if(rr<=limit)break;
            vcycle(list,red,black,coef,r,z,q,m1,n1,m2,n2,t,T);
            local=0.;FOR(n){int i=fineCell(list,red,k);local+=r[i]*z[i];}
            float rzNew=groupSum(local,partial,lane,warp,warps),beta=rzNew/max(rz,1e-30f);rz=rzNew;
            FOR(n){int i=fineCell(list,red,k);d[i]=z[i]+beta*d[i];}
            SYNC;
        }
    }
    if(t==0) {
        atomic_fetch_add_explicit(stats+49,uint(iterations),memory_order_relaxed);
        atomic_fetch_max_explicit(stats+50,uint(iterations),memory_order_relaxed);
        atomic_fetch_add_explicit(stats+51,1,memory_order_relaxed);
        atomic_fetch_max_explicit(stats+52,uint(min(1e9f,sqrt(rr/max(1.f,float(n)))/scale*1e6)),memory_order_relaxed);
    }
}
#undef FOR
#undef SYNC
kernel void fluidProject(device float4 *v [[buffer(4)]],device const int *type [[buffer(6)]],
                         device const float *pressure [[buffer(8)]],device const float *s [[buffer(10)]],
                         device const uint *active [[buffer(3)]],device const float4 *open [[buffer(25)]],
                         device float4 *solver [[buffer(29)]],
                         device const CupBody &body [[buffer(14)]],uint i [[thread_position_in_grid]]) {
    if(i>=CELLS||!active[i])return;int3 c=cellCoord(i);float3 vel=v[i].xyz;
    device const float *phi=solverVector(solver,0);
    for(int a=0;a<3;a++) {
        int3 q=c;q[a]--;
        if(blocked(c,a,open)){vel[a]=solidFace(c,a,body)[a];continue;}
        bool lq=inGrid(q)&&solveCell(cellIndex(q),type),li=solveCell(i,type);
        if(!lq&&!li)continue;
        // Ghost fluid: air pressure is zero at the interpolated surface, not
        // at the air cell's centre, so waves are not snapped to cells.
        float gradient;
        if(lq&&li)gradient=pressure[i]-pressure[cellIndex(q)];
        else if(li)gradient=pressure[i]/(inGrid(q)?ghostTheta(phi[i],phi[cellIndex(q)]):1.);
        else gradient=-pressure[cellIndex(q)]/ghostTheta(phi[cellIndex(q)],phi[i]);
        vel[a]-=gradient*s[0]/DX;
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
    return solveCell(cellIndex(c),type) || (inGrid(q)&&solveCell(cellIndex(q),type));
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
// Anisotropic (Yu-Turk) surface kernels. Per particle: neighbour covariance ->
// principal axes; the kernel keeps its in-plane radius but narrows across
// airborne sheets, so a spill sheet is averaged along its plane rather than
// following every particle's bump. Centres are Laplacian-smoothed toward the
// neighbour mean (bounded shift), which removes per-particle jitter. Written
// into the solver buffer, dead between substeps: 3 float4 per slot.
constant float KERNEL_RADIUS=DX*1.25;
constant float CENTRE_SMOOTHING=.9;
constant float CENTRE_SHIFT=.012;
void eigenSymmetric(float3x3 a,thread float3 &e,thread float3x3 &v) {
    v=float3x3(1.);
    for(int sweep=0;sweep<5;sweep++)for(int k=0;k<3;k++) {
        int p=k==2?1:0,q=k==0?1:2;float apq=a[q][p];
        if(abs(apq)<1e-14)continue;
        float theta=(a[q][q]-a[p][p])/(2.*apq);
        float t=(theta>=0.?1.:-1.)/(abs(theta)+sqrt(theta*theta+1.));
        float c=rsqrt(t*t+1.),s=t*c;
        float3x3 j=float3x3(1.);j[p][p]=c;j[q][q]=c;j[q][p]=s;j[p][q]=-s;
        a=transpose(j)*a*j;v=v*j;
    }
    e=float3(a[0][0],a[1][1],a[2][2]);
}
kernel void fluidAnisotropy(device const FluidParticle *particles [[buffer(0)]],device const int *heads [[buffer(1)]],
                            device const int *next [[buffer(2)]],device atomic_uint *details [[buffer(16)]],
                            device float4 *solver [[buffer(29)]],uint i [[thread_position_in_grid]]) {
    if(i>=atomic_load_explicit(details,memory_order_relaxed))return;
    FluidParticle p=particles[i];int3 c=particleCell(p.x.xyz);
    if(!particleAlive(p)||p.cx.w>0.||!inGrid(c)){solver[3*i]=0.;return;} // .w = 0: not in the field
    float3 sum=0.;float3x3 outer=float3x3(0.);float total=0.;int count=0;
    for(int z=-1;z<=1;z++)for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++) {
        int3 q=c+int3(x,y,z);if(!inGrid(q))continue;
        for(int j=heads[cellIndex(q)];j>=0;j=next[j]) {
            FluidParticle o=particles[j];if(o.cx.w>0.||int(i)==j)continue;
            float3 d=o.x.xyz-p.x.xyz;float r=length(d)/DX;if(r>=1.)continue;
            float w=(1.-r*r*r)*particleWeight(o);
            sum+=w*d;outer+=w*float3x3(d*d.x,d*d.y,d*d.z);total+=w;count++;
        }
    }
    float3 radii=KERNEL_RADIUS;float3x3 axes=float3x3(1.);float3 centre=p.x.xyz;
    if(count>=4) {
        float3 mean=sum/total,shift=CENTRE_SMOOTHING*mean*total/(total+particleWeight(p));
        centre+=shift*min(1.f,CENTRE_SHIFT/max(length(shift),1e-6f));
        if(count>=10) {
            float3x3 covariance=outer*(1./total)-float3x3(mean*mean.x,mean*mean.y,mean*mean.z);
            float3 e;eigenSymmetric(covariance,e,axes);
            float3 sigma=sqrt(max(e,float3(1e-12)));
            // Only genuinely planar, airborne neighbourhoods (pour sheets):
            // narrowing the bulk surface or a table puddle (half-space
            // neighbourhoods) only roughens it or opens holes.
            float3 ratio=sigma/max(sigma.x,max(sigma.y,sigma.z));
            if(min(ratio.x,min(ratio.y,ratio.z))<.45 && fixedSolid(p.x.xyz)>.05)radii=KERNEL_RADIUS*clamp(ratio,.6f,1.f);
        }
    }
    // G = R diag(1/r) R^T (symmetric): kernel argument s = |G (x - centre)|.
    float3x3 g=axes*float3x3(float3(1./radii.x,0,0),float3(0,1./radii.y,0),float3(0,0,1./radii.z))*transpose(axes);
    // Everything fluidSurface needs, so its gather never touches the particles.
    // Weight scaled by the kernel's volume ratio, so a narrowed kernel still
    // carries its particle's full share into the sheet-thickness heuristics.
    solver[3*i]=float4(centre,particleWeight(p)*KERNEL_RADIUS*KERNEL_RADIUS*KERNEL_RADIUS/(radii.x*radii.y*radii.z));
    solver[3*i+1]=float4(g[0][0],g[1][1],g[2][2],g[1][0]);
    solver[3*i+2]=float4(g[2][0],g[2][1],particleRadius(p),p.v.w);
}
// Covariance-aware moving-least-squares reconstruction over the anisotropic
// kernels. A sparse occupancy mask skips empty space; local normal-direction
// variance thins sheets/puddles without forcing liquid back into the cup or
// deleting mass. The raw distance goes to .w; fluidSurfaceSmooth filters and
// blends it into .x (the renderer's signed distance).
kernel void fluidSurface(device const FluidParticle *particles [[buffer(0)]],device const int *heads [[buffer(1)]],
                         device const int *next [[buffer(2)]],device const uint *active [[buffer(3)]],
                         device float4 *field [[buffer(11)]],device float4 *solver [[buffer(29)]],
                         uint i [[thread_position_in_grid]]) {
    if(i>=uint(SURFACE.x*SURFACE.y*SURFACE.z))return;
    int3 fine=int3(i%SURFACE.x,(i/SURFACE.x)%SURFACE.y,i/(SURFACE.x*SURFACE.y));
    float3 p=ORIGIN+(float3(fine)+.5)*SURFACE_DX;int3 c=particleCell(p);
    if(!inGrid(c)||!active[cellIndex(c)]){field[i]=float4(.12,0,0,.12);return;}
    float3 center=0.,diagonal=0.,off=0.;float total=0.,milk=0.,milkWeight=0.,coarseMilk=0.,nearest=.12,totalRadius=0.;
    for(int z=-1;z<=1;z++)for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++) {
        int3 q=c+int3(x,y,z);if(!inGrid(q))continue;
        for(int j=heads[cellIndex(q)];j>=0;j=next[j]) {
            float4 a=solver[3*j];if(a.w<=0.)continue;
            float4 b=solver[3*j+1],e=solver[3*j+2];
            float3 delta=a.xyz-p;float r2=dot(delta,delta);
            nearest=min(nearest,sqrt(r2)-e.z);
            float3 u=float3(b.x*delta.x+b.w*delta.y+e.x*delta.z,b.w*delta.x+b.y*delta.y+e.y*delta.z,e.x*delta.x+e.y*delta.y+b.z*delta.z);
            float w=1.-dot(u,u);w=w>0.?w*w*w*a.w:0.;
            totalRadius+=e.z*w;
            center+=delta*w;diagonal+=delta*delta*w;
            off+=float3(delta.x*delta.y,delta.x*delta.z,delta.y*delta.z)*w;
            float mw=1.-r2/(.06*.06);mw=mw>0.?mw*mw*mw*a.w:0.;
            milk+=e.w*mw;milkWeight+=mw;coarseMilk+=e.w*w;total+=w;
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
    field[i]=float4(field[i].x,concentration,total,clamp(phi,-.1f,.12f));
}
// One band-limited Laplacian pass (|change| <= half a voxel, so sheets thin
// but do not tear), then motion-adaptive temporal blending with the previous
// frame: sub-voxel jitter is averaged away, real motion passes straight through.
kernel void fluidSurfaceSmooth(device float4 *field [[buffer(11)]],device const uint *active [[buffer(3)]],
                               device const float *s [[buffer(10)]],uint i [[thread_position_in_grid]]) {
    if(i>=uint(SURFACE.x*SURFACE.y*SURFACE.z))return;
    int3 fine=int3(i%SURFACE.x,(i/SURFACE.x)%SURFACE.y,i/(SURFACE.x*SURFACE.y));
    int3 c=particleCell(ORIGIN+(float3(fine)+.5)*SURFACE_DX);
    if(!inGrid(c)||!active[cellIndex(c)])return; // fluidSurface already wrote .12
    float raw=field[i].w,phi=raw;
    if(abs(raw)<3.*SURFACE_DX) {
        float sum=0.;
        for(int a=0;a<3;a++)for(int direction=-1;direction<=1;direction+=2) {
            int3 q=fine;q[a]=clamp(q[a]+direction,0,SURFACE[a]-1);
            sum+=field[(q.z*SURFACE.y+q.y)*SURFACE.x+q.x].w;
        }
        phi+=clamp(.5*(sum/6.-raw),-.5f*SURFACE_DX,.5f*SURFACE_DX);
    }
    if(s[24]>0.) {
        float old=field[i].x,change=abs(phi-old);
        phi=mix(phi,old,.5*(1.-smoothstep(.25f*SURFACE_DX,1.5f*SURFACE_DX,change)));
    }
    field[i].x=clamp(phi,-.1f,.12f);
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
                           device const float4 *mass [[buffer(5)]],device const float4 *open [[buffer(25)]],
                           device const float *divergence [[buffer(7)]],device atomic_uint *stats [[buffer(12)]],
                           device float4 *solver [[buffer(29)]],
                           device const CupBody &body [[buffer(14)]],uint i [[thread_position_in_grid]]) {
    if(i>=CELLS||type[i]!=2)return;
    int3 c=cellCoord(i);
    // Unweighted divergence over every particle cell (the pre-cut-cell metric).
    atomic_fetch_add_explicit(stats+53,uint(min(100000.f,abs(divergenceAt(c,v))*1000.)),memory_order_relaxed);
    atomic_fetch_add_explicit(stats+54,1,memory_order_relaxed);
    if(solveCell(i,type)) {
        atomic_fetch_add_explicit(stats+11,uint(min(100000.f,abs(divergence[i])*1000.)),memory_order_relaxed);
        atomic_fetch_add_explicit(stats+12,uint(min(100000.f,abs(cutDivergence(c,v,open,body))*1000.)),memory_order_relaxed);
        atomic_fetch_add_explicit(stats+13,1,memory_order_relaxed);
    }
    atomic_fetch_max_explicit(stats+27,uint(mass[i].w*1000.),memory_order_relaxed);
    atomic_fetch_add_explicit(stats+28,1,memory_order_relaxed);
    if(mass[i].w>10.)atomic_fetch_add_explicit(stats+29,1,memory_order_relaxed);
    atomic_fetch_add_explicit(stats+30,uint(min(1.f,mass[i].w/8.)*DX*DX*DX*1e6),memory_order_relaxed);
}

// Integrate fluid pressure traction on the moving shell. Particle collision
// impulses account for contacts below grid resolution; this handles bulk load.
kernel void fluidReaction(device const int *type [[buffer(6)]],device const float *pressure [[buffer(8)]],
                          device const float *s [[buffer(10)]],device const CupBody &body [[buffer(14)]],
                          device const float4 *open [[buffer(25)]],device float4 *solver [[buffer(29)]],
                          device atomic_int *reaction [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=CELLS || body.position.w<.5 || !solveCell(i,type))return;
    int3 c=cellCoord(i);
    float density=.25/(s[18]*.0425*.0425*.0425);
    for(int a=0;a<3;a++)for(int direction=-1;direction<=1;direction+=2) {
        int3 face=c;if(direction>0)face[a]++;
        // The closed share of a cut face carries the traction.
        float closed=inGrid(face)?1.-open[cellIndex(face)][a]:0.;
        if(closed<=0.)continue;
        float3 p=ORIGIN+(float3(face)+faceOffset(a))*DX;
        if(!isCupContact(p,body))continue;
        float3 impulse=0.;
        impulse[a]=float(direction)*clamp(pressure[i],0.f,100.f)*DX*DX*s[0]*density*closed;
        accumulateReaction(reaction,impulse,p-body.position.xyz);
    }
}
