// Conservative sub-grid splash and wet-surface detail. Fixed-point liquid
// transfers are exact; stains are transported nonvolatile visual tracers.
uint reserveSlots(device atomic_uint *counter,uint count) {
    uint old=atomic_load_explicit(counter,memory_order_relaxed);
    while(old+count<=SLOT_CAPACITY) {
        if(atomic_compare_exchange_weak_explicit(counter,&old,old+count,
                                                memory_order_relaxed,memory_order_relaxed))return old;
    }
    return SLOT_CAPACITY;
}
kernel void fluidBreakup(device FluidParticle *particles [[buffer(0)]],
                          device const float4 *mass [[buffer(5)]],device const float *s [[buffer(10)]],
                          device const CupBody &body [[buffer(14)]],device atomic_uint *details [[buffer(16)]],
                          uint i [[thread_position_in_grid]]) {
    if(i>=uint(s[4]))return;
    FluidParticle p=particles[i];
    if(!particleAlive(p)||p.x.w!=8.||p.cx.w>0.||inCup(p.x.xyz,body))return;
    int3 c=particleCell(p.x.xyz);if(!inGrid(c))return;
    float density=mass[cellIndex(c)].w,speed=length(p.v.xyz);
    float3 local=cupLocal(p.x.xyz,body);
    bool rim=local.y>.78 && local.y<1.13 && length(local.xz)>.75;
    // Resolve only sparse fast free edges, not the bulk or resting surface.
    if(density>5. || speed<(rim?1.2:3.) || solidDistance(p.x.xyz,body)<.048)return;
    float3 affine[3]={p.cx.xyz,p.cy.xyz,p.cz.xyz};
    float3 sumVelocity=0.;
    if(density>1.5) {
        // Resolvable sheet (an over-the-rim pour): reseed only where it is
        // stretching, as four on-grid quarters spread along the two stretching
        // axes of the APIC strain, so the sheet thins but stays continuous.
        float3x3 gradient=transpose(float3x3(affine[0],affine[1],affine[2]));
        float3 e;float3x3 axes;eigenSymmetric(.5*(gradient+transpose(gradient)),e,axes);
        int k0=e.x>=e.y&&e.x>=e.z?0:(e.y>=e.z?1:2),k1=k0==0?(e.y>=e.z?1:2):(k0==1?(e.x>=e.z?0:2):(e.x>=e.y?0:1));
        if(e[k0]<4.||e[k1]<-2.)return;
        uint start=reserveSlots(details,3);if(start==SLOT_CAPACITY)return;
        for(uint child=0;child<4;child++) {
            float3 offset=axes[child<2?k0:k1]*((child&1)?.012:-.012);
            FluidParticle q=p;q.x=float4(p.x.xyz+offset,2.);
            q.v.xyz+=float3(dot(affine[0],offset),dot(affine[1],offset),dot(affine[2],offset));
            sumVelocity+=q.v.xyz;
            particles[child==0?i:start+child-1]=q;
        }
        sumVelocity*=2.;
    } else {
        if(i%5!=0 && !rim)return;
        uint start=reserveSlots(details,7);if(start==SLOT_CAPACITY)return;
        for(uint child=0;child<8;child++) {
            float3 offset=float3(child&1?1.:-1.,child&2?1.:-1.,child&4?1.:-1.)*.010;
            FluidParticle q=p;q.x=float4(p.x.xyz+offset,1.);
            // Symmetric samples of the parent's affine field preserve momentum.
            // Their fine-scale energy comes from the existing APIC affine motion.
            q.v.xyz+=float3(dot(affine[0],offset),dot(affine[1],offset),dot(affine[2],offset));
            q.cx=float4(0,0,0,.001);q.cy=0.;q.cz=0.;
            sumVelocity+=q.v.xyz;
            particles[child==0?i:start+child-1]=q;
        }
    }
    atomic_fetch_max_explicit(details+9,uint(length(sumVelocity/8.-p.v.xyz)*1e6),memory_order_relaxed);
    atomic_fetch_add_explicit(details+1,1,memory_order_relaxed);
    if(rim)atomic_fetch_add_explicit(details+8,1,memory_order_relaxed);
}

void depositFilm(thread FluidParticle &p,float impact,device const uint4 *film,
                 device atomic_uint *pending,device atomic_uint *details) {
    int2 c=filmCoord(p.x.xz);if(!inFilm(c)||p.x.w<=0)return;
    float base=filmBaseHeight(p.x.xz);
    if(p.x.y-base>.032 || p.x.y<base-.005)return;
    int index=filmIndex(c);
    // Keep deep pools in APIC. Only the thin boundary layer enters the film.
    uint maxQ=uint(.032*FILM_DX*FILM_DX/FILM_VOLUME_Q);
    uint existing=film[index].x;
    if(existing>=maxQ)return;
    device atomic_uint *water=pending+index*4;
    uint old=atomic_load_explicit(water,memory_order_relaxed),units=0;
    while(old+FILM_Q<=maxQ-existing) {
        units=min(uint(p.x.w),(maxQ-existing-old)/FILM_Q);
        if(!units)break;
        if(atomic_compare_exchange_weak_explicit(water,&old,old+units*FILM_Q,
                                                memory_order_relaxed,memory_order_relaxed))break;
        units=0;
    }
    if(!units)return;
    uint q=units*FILM_Q;
    atomic_fetch_add_explicit(pending+index*4+1,uint(float(q)*(1.-p.v.w)),memory_order_relaxed);
    atomic_fetch_add_explicit(pending+index*4+2,uint(float(q)*p.v.w),memory_order_relaxed);
    if(impact>.7)atomic_fetch_add_explicit(pending+index*4+3,uint(min(12.f,impact)*float(units)*100.),memory_order_relaxed);
    atomic_fetch_add_explicit(details+(base>-.12?2:3),units,memory_order_relaxed);
    atomic_fetch_add_explicit(details+4,uint(p.v.w*float(units)*8192.),memory_order_relaxed);
    if(impact>1.)atomic_fetch_add_explicit(details+6,1,memory_order_relaxed);
    p.x.w-=float(units);
    if(p.x.w<=0.){p.x.w=0.;p.cz.w=3.;}
}

kernel void filmGather(device atomic_uint *pending [[buffer(17)]],device uint4 *film [[buffer(18)]],
                       device float2 *ripples [[buffer(22)]],device const float *s [[buffer(10)]],uint i [[thread_position_in_grid]]) {
    if(i>=FILM_CELLS)return;
    uint4 add;
    for(int a=0;a<4;a++)add[a]=atomic_exchange_explicit(pending+i*4+a,0,memory_order_relaxed);
    film[i].xyz+=add.xyz;
    if(add.w) {
        ripples[i].y+=min(.05f,float(add.w)*.000012);
        film[i].w=0x80000000u | min(0x7ffffffeu,uint(s[5]*1000.)+1);
    }
}
int2 filmDirection(int a) {
    return a==0?int2(-1,0):(a==1?int2(1,0):(a==2?int2(0,-1):int2(0,1)));
}
bool filmBlocked(int2 c,CupBody body) {
    float2 p=filmCenter(c);
    return cupShape(cupLocal(float3(p.x,filmBaseHeight(p)+.002,p.y),body))<.001;
}
kernel void filmFlux(device const uint4 *film [[buffer(18)]],device uint4 *flux [[buffer(19)]],
                     device const float *s [[buffer(10)]],device const CupBody &body [[buffer(14)]],
                     uint i [[thread_position_in_grid]]) {
    if(i>=FILM_CELLS)return;
    int2 c=int2(i%FILM_N,i/FILM_N);uint water=film[i].x;flux[i]=0;
    if(!water||filmBlocked(c,body))return;
    float2 p=filmCenter(c);float depth=filmDepth(water),base=filmBaseHeight(p);
    float3 g=float3(s[1],s[2],s[3]);float gravity=max(1.f,length(g));
    float head=(base+depth)*(-g.y/gravity)-dot(g.xz/gravity,p);
    float4 fractions=0.;
    for(int a=0;a<4;a++) {
        int2 n=c+filmDirection(a);if(!inFilm(n)||filmBlocked(n,body))continue;
        float2 np=filmCenter(n);float nd=filmDepth(film[filmIndex(n)].x);
        float neighbor=(filmBaseHeight(np)+nd)*(-g.y/gravity)-dot(g.xz/gravity,np);
        // Shallow conservative spreading plus downhill flow; each directed
        // flux has one integer value, consumed identically at both endpoints.
        float difference=head-neighbor+.12*(depth-nd);
        fractions[a]=clamp(difference/max(.003f,depth+nd),0.f,1.f)*min(.2f,s[21]*9.);
    }
    float total=dot(fractions,float4(1));
    if(total>.6)fractions*=.6/total;
    uint4 outgoing=uint4(float(water)*fractions);
    uint cap=uint(.032*FILM_DX*FILM_DX/FILM_VOLUME_Q);
    for(int a=0;a<4;a++) {
        int2 n=c+filmDirection(a);
        if(!inFilm(n)){outgoing[a]=0;continue;}
        uint neighbor=film[filmIndex(n)].x;
        // Each of four neighbors receives at most a quarter of available room.
        // This keeps the reduced-dimensional reservoir genuinely thin.
        outgoing[a]=min(outgoing[a],(cap-min(cap,neighbor))/4);
    }
    flux[i]=outgoing;
}
kernel void filmAdvance(device const uint4 *film [[buffer(18)]],device const uint4 *flux [[buffer(19)]],
                        device uint4 *dry [[buffer(20)]],device uint4 *next [[buffer(21)]],
                        device const float2 *ripples [[buffer(22)]],device float2 *waveNext [[buffer(23)]],
                        device const float *s [[buffer(10)]],device const CupBody &body [[buffer(14)]],
                        uint i [[thread_position_in_grid]]) {
    if(i>=FILM_CELLS)return;
    int2 c=int2(i%FILM_N,i/FILM_N);uint4 state=film[i];uint3 value=state.xyz;
    float edge=0.,lap=0.;uint timestamp=state.w&0x7fffffffu;
    for(int a=0;a<4;a++) {
        int2 n=c+filmDirection(a);if(!inFilm(n))continue;
        int j=filmIndex(n);uint4 neighbor=film[j];
        uint out=flux[i][a],incoming=flux[j][a^1];
        uint2 outDye=state.x?uint2(float2(state.yz)*(float(out)/state.x)):uint2(0);
        uint2 inDye=neighbor.x?uint2(float2(neighbor.yz)*(float(incoming)/neighbor.x)):uint2(0);
        value-=uint3(out,outDye);value+=uint3(incoming,inDye);
        edge+=neighbor.x<128?1.:0.;
        if(neighbor.x>32 && abs(ripples[j].x)>1e-7)
            timestamp=max(timestamp,neighbor.w&0x7fffffffu);
        lap+=(neighbor.x>32 && !filmBlocked(n,body)?ripples[j].x:ripples[i].x)-ripples[i].x;
    }
    uint4 residue=dry[i];
    if(value.x) {
        float dt=s[21],depth=filmDepth(value.x);
        // Edge-enhanced evaporation and tracer pinning yield irregular rings
        // from the actual wet footprint, never stamped decorative circles.
        float rate=.000018+.000065*edge;
        if(filmBlocked(c,body))rate*=.12;
        float evaporation=rate*dt*FILM_DX*FILM_DX/FILM_VOLUME_Q+float(residue.w)/65536.;
        uint evaporated=min(value.x,uint(evaporation));
        residue.w=uint(fract(evaporation)*65536.);
        value.x-=evaporated;residue.z+=evaporated;
        float pin=min(.25f,dt*edge*1.6)*(1.-smoothstep(.001f,.008f,depth));
        uint2 pinned=value.x?uint2(float2(value.yz)*pin):value.yz;
        residue.xy+=pinned;value.yz-=pinned;
    } else {residue.xy+=value.yz;value.yz=0;}
    uint footprint=(state.w&0x80000000u) | (value.x>32?0x80000000u:0u);
    next[i]=uint4(value,footprint|timestamp);dry[i]=residue;
    float2 wave=ripples[i];
    float age=timestamp?max(0.f,s[5]-float(timestamp-1)*.001):1000.;
    if(value.x<32 || filmBlocked(c,body) || age>3.)wave=0.;
    else {
        float dt=s[21];
        wave.y=(wave.y+lap*(.75*.75/(FILM_DX*FILM_DX))*dt)*exp(-dt*6.);
        // A moving wet boundary must not sustain numerical ringing forever.
        // Timestamp propagation ties the sub-grid ripple envelope to impacts.
        float envelope=exp(-max(0.f,age-.4)*2.);
        wave.y=clamp(wave.y,-.1*envelope,.1*envelope);
        wave.x=clamp(wave.x+wave.y*dt,-.0015*envelope,.0015*envelope);
    }
    waveNext[i]=wave;
}
kernel void filmCommit(device uint4 *film [[buffer(18)]],device const uint4 *next [[buffer(21)]],
                       device float2 *ripples [[buffer(22)]],device const float2 *waveNext [[buffer(23)]],
                       uint i [[thread_position_in_grid]]) {
    if(i>=FILM_CELLS)return;film[i]=next[i];ripples[i]=waveNext[i];
}
kernel void filmStats(device const uint4 *film [[buffer(18)]],device const uint4 *dry [[buffer(20)]],
                      device const float2 *ripples [[buffer(22)]],device atomic_uint *stats [[buffer(12)]],
                      uint i [[thread_position_in_grid]]) {
    if(i>=FILM_CELLS)return;
    atomic_fetch_max_explicit(stats+46,film[i].x,memory_order_relaxed);
    atomic_fetch_add_explicit(stats+32,film[i].x,memory_order_relaxed);
    atomic_fetch_add_explicit(stats+33,dry[i].z,memory_order_relaxed);
    atomic_fetch_add_explicit(stats+34,film[i].y+dry[i].x,memory_order_relaxed);
    atomic_fetch_add_explicit(stats+35,film[i].z+dry[i].y,memory_order_relaxed);
    if(film[i].x>32)atomic_fetch_add_explicit(stats+36,1,memory_order_relaxed);
    if(dry[i].x+dry[i].y>32)atomic_fetch_add_explicit(stats+37,1,memory_order_relaxed);
    if(film[i].w&0x80000000u)atomic_fetch_add_explicit(stats+44,1,memory_order_relaxed);
    int2 c=int2(i%FILM_N,i/FILM_N);bool perimeter=false;
    for(int a=0;a<4;a++) {
        int2 n=c+filmDirection(a);
        if(inFilm(n)&&(film[filmIndex(n)].w&0x80000000u)==0)perimeter=true;
    }
    if(perimeter)atomic_fetch_add_explicit(stats+45,dry[i].x+dry[i].y,memory_order_relaxed);
    atomic_fetch_add_explicit(stats+38,dry[i].x,memory_order_relaxed);
    atomic_fetch_add_explicit(stats+39,dry[i].y,memory_order_relaxed);
    atomic_fetch_max_explicit(stats+42,uint(abs(ripples[i].x)*1e8),memory_order_relaxed);
}

float4 sampleFilm(float2 xz,device const uint4 *film,device const uint4 *dry) {
    float2 g=(xz-FILM_ORIGIN)/FILM_DX-.5;int2 c=int2(floor(g));float2 f=fract(g);
    float4 sample=0.;
    for(int y=0;y<2;y++)for(int x=0;x<2;x++) {
        int2 n=c+int2(x,y);if(!inFilm(n))continue;
        int i=filmIndex(n);float w=(x?f.x:1.-f.x)*(y?f.y:1.-f.y);
        uint4 a=film[i],b=dry[i];
        sample+=float4(filmDepth(a.x),float(a.y+b.x),float(a.z+b.y),float(b.x+b.y))*w;
    }
    // yz are areal tracer load in particle-equivalent thickness units.
    sample.yzw*=FILM_VOLUME_Q/(FILM_DX*FILM_DX);
    return sample;
}
float2 wetHeight(int2 c,device const uint4 *film,device const float2 *ripples) {
    if(!inFilm(c))return 0.;int i=filmIndex(c);
    return float2(min(.012f,filmDepth(film[i].x)),ripples[i].x);
}
// Central-difference slopes per film cell, bilinearly blended: C0 normals
// instead of one constant facet per 2 cm cell. Depth and ripple slopes are
// kept apart so thin edges become a rounded meniscus while ripples stay soft.
float3 wetNormal(float3 p,float3 n,device const uint4 *film,device const float2 *ripples) {
    float2 g=(p.xz-FILM_ORIGIN)/FILM_DX-.5;int2 c=int2(floor(g));float2 f=fract(g);
    float2 h[4][4];
    for(int y=0;y<4;y++)for(int x=0;x<4;x++)h[y][x]=wetHeight(c+int2(x-1,y-1),film,ripples);
    float4 slope=0.;float depth=0.;
    for(int y=1;y<3;y++)for(int x=1;x<3;x++) {
        float w=(x==2?f.x:1.-f.x)*(y==2?f.y:1.-f.y);
        slope+=w*float4(h[y][x+1]-h[y][x-1],h[y+1][x]-h[y-1][x]).xzyw;
        depth+=w*h[y][x].x;
    }
    slope/=2.*FILM_DX;
    // xy: depth slope (meniscus at the wet edge), zw: low-passed ripple slope.
    float2 meniscus=clamp(slope.xy*.35,-.22f,.22f),ripple=clamp(slope.zw*.30,-.10f,.10f);
    float2 gradient=(meniscus+ripple)*smoothstep(.00005f,.0012f,depth);
    return normalize(n-float3(gradient.x,0,gradient.y));
}
// Tricubic B-spline gradient of the reconstructed liquid distance. Trilinear
// field samples are only C0, so their differences facet along voxel planes.
float3 fluidGradient(float3 p,device const float *raw) {
    device const float4 *field=reinterpret_cast<device const float4 *>(raw);
    float3 g=(p-ORIGIN)/SURFACE_DX-.5;int3 c=int3(floor(g));float3 f=g-floor(g);
    float3 w[4],d[4];
    for(int k=0;k<3;k++) {
        float t=f[k],s=1.-t;
        w[0][k]=s*s*s/6.;w[1][k]=(3.*t*t*t-6.*t*t+4.)/6.;w[2][k]=(-3.*t*t*t+3.*t*t+3.*t+1.)/6.;w[3][k]=t*t*t/6.;
        d[0][k]=-.5*s*s;d[1][k]=1.5*t*t-2.*t;d[2][k]=-1.5*t*t+t+.5;d[3][k]=.5*t*t;
    }
    float3 gradient=0.;
    for(int z=0;z<4;z++)for(int y=0;y<4;y++) {
        int cz=clamp(c.z+z-1,0,SURFACE.z-1),cy=clamp(c.y+y-1,0,SURFACE.y-1);
        int row=(cz*SURFACE.y+cy)*SURFACE.x;
        float3 a=0.;
        for(int x=0;x<4;x++) {
            float v=field[row+clamp(c.x+x-1,0,SURFACE.x-1)].x;
            a+=v*float3(d[x].x,w[x].x,w[x].x);
        }
        gradient+=a*float3(w[y].y*w[z].z,d[y].y*w[z].z,w[y].y*d[z].z);
    }
    return gradient/SURFACE_DX;
}
float3 stainColor(float3 base,float4 sample) {
    float share=sample.y/max(1e-8f,sample.y+sample.z);
    float coffee=1.-exp(-(sample.y+sample.w*share*5.)*110.);
    float cream=1.-exp(-(sample.z+sample.w*(1.-share)*3.)*100.);
    float dry=1.-smoothstep(.00015f,.002f,sample.x);
    float3 stained=base*mix(float3(1),float3(.40,.19,.08),coffee*(.55+.35*dry));
    return mix(stained,float3(.48,.36,.21),cream*.55);
}

SprayHit traceSpray(float3 ro,float3 rd,float maxT,device const FluidParticle *particles,
                    device const int *heads,device const int *next,device const uint *active) {
    SprayHit hit;hit.distance=maxT;hit.normal=0.;hit.cream=0.;hit.radius=0.;
    float3 inv=1./select(rd,float3(1e-8),abs(rd)<1e-8);
    float3 a=(ORIGIN-ro)*inv,b=(ORIGIN+float3(GRID)*DX-ro)*inv;
    float3 low=min(a,b),high=max(a,b);
    float t=max(0.f,max(low.x,max(low.y,low.z))),end=min(maxT,min(high.x,min(high.y,high.z)));
    if(t>=end)return hit;
    int3 c=clamp(particleCell(ro+rd*(t+.0001)),int3(0),GRID-1);
    int3 step=int3(rd.x>=0?1:-1,rd.y>=0?1:-1,rd.z>=0?1:-1);
    float3 edge=ORIGIN+(float3(c)+float3(step.x>0,step.y>0,step.z>0))*DX;
    float3 crossing=(edge-ro)*inv,delta=abs(DX*inv);
    for(int iteration=0;iteration<180 && inGrid(c) && t<end && t<hit.distance;iteration++) {
        if(active[cellIndex(c)]&2) {
            for(int z=-1;z<=1;z++)for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++) {
                int3 bucket=c+int3(x,y,z);if(!inGrid(bucket))continue;
                for(int j=heads[cellIndex(bucket)];j>=0;j=next[j]) {
                    FluidParticle p=particles[j];if(p.cx.w<=0.)continue;
                    float radius=particleRadius(p);float3 q=ro-p.x.xyz;
                    float projection=dot(q,rd),disc=projection*projection-dot(q,q)+radius*radius;
                    if(disc<=0.)continue;
                    float d=-projection-sqrt(disc);
                    if(d>0. && d<hit.distance) {
                        hit.distance=d;hit.normal=normalize(ro+rd*d-p.x.xyz);hit.cream=p.v.w;hit.radius=radius;
                    }
                }
            }
        }
        int axis=crossing.x<crossing.y?(crossing.x<crossing.z?0:2):(crossing.y<crossing.z?1:2);
        t=crossing[axis];crossing[axis]+=delta[axis];c[axis]+=step[axis];
    }
    return hit;
}
