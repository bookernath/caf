// Focused GPU conservation/decay fixtures using the production kernels.
import Foundation
import Metal
let root=URL(fileURLWithPath:CommandLine.arguments[1])
let mode=CommandLine.arguments.count>2 ? CommandLine.arguments[2]:"ripple"
guard let device=MTLCreateSystemDefaultDevice(),let queue=device.makeCommandQueue() else {fatalError("Metal unavailable")}
let files=["caf-scene.metal","caf.metal","caf-fluid.metal","caf-detail.metal"]
let source=try "#define CAF_SCENE_LAYOUT 5\n"+files.map {try String(contentsOf:root.appendingPathComponent($0),encoding:.utf8)}.joined(separator:"\n")
let library=try device.makeLibrary(source:source,options:nil)
let n=384*384,grid=56*32*56
var sizes=[Int](repeating:256,count:25)
sizes[0]=120000*80;sizes[5]=grid*16
for i in 17...21 {sizes[i]=n*16}
sizes[22]=n*8;sizes[23]=n*8
let buffers=sizes.map {size -> MTLBuffer in
    let b=device.makeBuffer(length:size,options:.storageModeShared)!;memset(b.contents(),0,size);return b
}
var params=[Float](repeating:0,count:32);params[2] = -98.1;params[21] = 1/90;params[0] = 1/360
params[4]=1;params[18]=15032
let body=buffers[14].contents().bindMemory(to:Float.self,capacity:20)
body[0]=10;body[1]=0.52;body[7]=1 // move cup away from the film fixture
var pipelines:[String:MTLComputePipelineState]=[:]
for name in ["fluidBreakup","filmFlux","filmAdvance","filmCommit"] {
    pipelines[name]=try device.makeComputePipelineState(function:library.makeFunction(name:name)!)
}
func dispatch(_ name:String,_ count:Int,_ encoder:MTLComputeCommandEncoder) {
    let pipeline=pipelines[name]!
    encoder.setComputePipelineState(pipeline)
    for i in 0..<buffers.count where i != 10 {encoder.setBuffer(buffers[i],offset:0,index:i)}
    encoder.setBytes(params,length:128,index:10)
    encoder.dispatchThreads(MTLSize(width:count,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:128,height:1,depth:1))
    encoder.memoryBarrier(scope:.buffers)
}
func run(_ iterations:Int,_ operations:[String],_ count:Int=n) {
    let command=queue.makeCommandBuffer()!,encoder=command.makeComputeCommandEncoder()!
    for _ in 0..<iterations {params[5]+=params[21];for name in operations {dispatch(name,count,encoder)}}
    encoder.endEncoding();command.commit();command.waitUntilCompleted()
    guard command.status == .completed else {fatalError(command.error!.localizedDescription)}
}
var output:[String:Any]=[:]
if mode=="split" {
    let particles=buffers[0].contents().bindMemory(to:Float.self,capacity:120000*20)
    let counter=buffers[16].contents().bindMemory(to:UInt32.self,capacity:32)
    let original:[Float]=[1.8,1.4,0,8, 2,1,-4,0.3, 1,2,3,0, -2,3,1,0, 1,-2,2,0]
    for i in 0..<20 {particles[i]=original[i]};counter[0]=1
    run(1,["fluidBreakup"],1)
    let count=Int(counter[0]);var volume:Float=0;var momentum=[Float](repeating:0,count:3)
    for i in 0..<count {
        let weight=particles[i*20+3]/8;volume+=weight
        for a in 0..<3 {momentum[a]+=weight*particles[i*20+4+a]}
    }
    output=["slots":count,"volume":volume,"momentum":momentum,"split_events":counter[1],"momentum_error":counter[9]]
    for i in 0..<20 {particles[i]=original[i]};counter[0]=119997
    run(1,["fluidBreakup"],1)
    output["capacity_slots"]=counter[0];output["unsplit_weight"]=particles[3]
} else {
    let water=buffers[18].contents().bindMemory(to:UInt32.self,capacity:n*4)
    let dry=buffers[20].contents().bindMemory(to:UInt32.self,capacity:n*4)
    let waves=buffers[22].contents().bindMemory(to:Float.self,capacity:n*2)
    var initial:UInt64=0
    for y in 0..<384 {for x in 0..<384 {
        let i=y*384+x
        if mode=="ripple" || (x-280)*(x-280)+(y-280)*(y-280)<64 {
            water[i*4]=mode=="ripple" ? 4096:512;water[i*4+1]=water[i*4];water[i*4+3]=0x80000000
            initial+=UInt64(water[i*4])
        }
    }}
    if mode=="ripple" {waves[(280*384+280)*2+1]=0.05;water[(280*384+280)*4+3]|=1}
    run(mode=="ripple" ? 45:900,mode=="ripple" ? ["filmAdvance","filmCommit"]:["filmFlux","filmAdvance","filmCommit"])
    var peak:Float=0
    for i in 0..<n {peak=max(peak,abs(waves[i*2]))}
    if mode=="ripple" {run(855,["filmAdvance","filmCommit"])}
    var final:UInt64=0,evap:UInt64=0,pigment:UInt64=0,residue:UInt64=0
    var maxWave:Float=0,maxVelocity:Float=0,wet=0,stained=0
    for i in 0..<n {
        final+=UInt64(water[i*4]);evap+=UInt64(dry[i*4+2]);pigment+=UInt64(water[i*4+1])+UInt64(dry[i*4])
        residue+=UInt64(dry[i*4]);if water[i*4]>32 {wet+=1};if dry[i*4]>32 {stained+=1}
        maxWave=max(maxWave,abs(waves[i*2]));maxVelocity=max(maxVelocity,abs(waves[i*2+1]))
    }
    output=["initial":initial,"water":final,"evaporated":evap,"pigment":pigment,"residue":residue,
            "wet_cells":wet,"stained_cells":stained,"peak":peak,"final_wave":maxWave,"final_velocity":maxVelocity]
}
let json=try JSONSerialization.data(withJSONObject:output,options:[.sortedKeys]);print(String(data:json,encoding:.utf8)!)
