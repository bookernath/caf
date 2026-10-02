// Persistent Metal renderer + APIC simulation. stdin: bounded JSON lines.
// stdout: CAF_METAL_5\n, then metadata length + JSON + RGB length + RGB bytes.
// No particle positions are read back during interactive rendering.
import Foundation
import Metal

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("caf-metal: " + message + "\n").utf8))
    exit(1)
}
func floats(_ value: Any?, count: Int? = nil) -> [Float]? {
    guard let values = value as? [NSNumber] else { return nil }
    let result = values.map { $0.floatValue }
    guard result.allSatisfy({ $0.isFinite }), count == nil || result.count == count else { return nil }
    return result
}
func writePacket(_ data: Data) {
    var count = UInt32(data.count).littleEndian
    withUnsafeBytes(of: &count) { FileHandle.standardOutput.write(Data($0)) }
    FileHandle.standardOutput.write(data)
}
guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    fail("Metal unavailable")
}
let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
let shaderURL = CommandLine.arguments.count > 1 ? URL(fileURLWithPath: CommandLine.arguments[1])
    : executable.deletingLastPathComponent().appendingPathComponent("caf.metal")
var pipelines: [String: MTLComputePipelineState] = [:]
do {
    let fluidURL = shaderURL.deletingLastPathComponent().appendingPathComponent("caf-fluid.metal")
    let sceneURL = shaderURL.deletingLastPathComponent().appendingPathComponent("caf-scene.metal")
    let detailURL = shaderURL.deletingLastPathComponent().appendingPathComponent("caf-detail.metal")
    let source = try "#define CAF_SCENE_LAYOUT 5\n" + String(contentsOf: sceneURL, encoding: .utf8) + "\n" + String(contentsOf: shaderURL, encoding: .utf8) + "\n" + String(contentsOf: fluidURL, encoding: .utf8) + "\n" + String(contentsOf: detailURL, encoding: .utf8)
    let library = try device.makeLibrary(source: source, options: nil)
    for name in ["coffee", "fluidClear", "fluidBins", "fluidP2G", "fluidForces", "fluidDivergence",
                 "fluidPressure", "fluidProject", "fluidIndirect", "fluidVorticity", "fluidExtrapolate", "fluidG2P", "fluidEvents", "fluidSurface", "fluidStats", "fluidGridStats", "cupEvents", "cupStep", "fluidReaction", "fluidBreakup", "filmGather", "filmFlux", "filmAdvance", "filmCommit", "filmStats"] {
        guard let function = library.makeFunction(name: name) else { fail("missing kernel \(name)") }
        pipelines[name] = try device.makeComputePipelineState(function: function)
    }
    if pipelines["fluidPressure"]!.maxTotalThreadsPerThreadgroup < 128 { fail("fluidPressure needs 128-wide threadgroups") }
} catch { fail(String(describing: error)) }

final class Fluid {
    static let cells = 56 * 32 * 56, surfaceCells = 224 * 128 * 224, capacity = 60000, slots = 120000, filmCells = 384*384
    let buffers: [MTLBuffer]
    var allocated = 0, initial = 0, fullCount = 0, emitted = 0, pourRemaining = 0
    var creamRemaining = 0, creamEmitted = 0
    var time: Float = 0
    var speedReference: Float = 18 // conservative until the first stats arrive
    var lastSteps = 0
    var metadata: [String: Any] = [:]
    var surfaceDirty = true
    init(level: Float, device: MTLDevice) {
        let n = Fluid.cells
        let f = Fluid.filmCells
        let sizes = [Fluid.slots * 80, n*4, Fluid.slots*4, n*4, n*16, n*16,
                     n*4, n*4, n*4, n*4, 128, Fluid.surfaceCells*16, 256, 16, 80, 64,
                     128, f*16, f*16, f*16, f*16, f*16, f*8, f*8, 64*16,
                     n*16, n*16, n*4, 64] // 25 face open fractions, 26 scratch, 27 red/black liquid lists, 28 counts + indirect args
        buffers = sizes.map { size in
            guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else { fail("fluid allocation failed") }
            memset(buffer.contents(), 0, size)
            return buffer
        }
        reset(level: level)
    }
    func reset(level: Float) {
        memset(buffers[0].contents(), 0, buffers[0].length)
        memset(buffers[8].contents(), 0, buffers[8].length)
        memset(buffers[14].contents(), 0, 80)
        memset(buffers[15].contents(), 0, 64)
        for i in 16..<buffers.count {memset(buffers[i].contents(),0,buffers[i].length)}
        let body = buffers[14].contents().bindMemory(to:Float.self, capacity:20)
        body[1]=0.52;body[3]=1;body[7]=1
        allocated = 0; fullCount = 0; emitted = 0; pourRemaining = 0; time = 0
        let points = buffers[0].contents().bindMemory(to: Float.self, capacity: Fluid.slots * 20)
        let spacing: Float = 0.0425
        // Eight particles per pressure-grid cell, seeded strictly inside the
        // ceramic. Reference full volume is independent of starting fill.
        for iy in 0..<19 {
            let y: Float = 0.165 + Float(iy)*spacing
            if y > 0.92 { continue }
            let outer: Float = y < 0.27 ? 0.66+(y-0.13)/0.14*0.09 : 0.75+(y-0.27)/0.73*0.05
            let radius = outer - 0.069
            for iz in -18...18 { for ix in -18...18 {
                let x = (Float(ix)+0.25)*spacing, z = (Float(iz)+0.25)*spacing
                if x*x+z*z >= radius*radius { continue }
                fullCount += 1
                if y > 0.30+0.62*max(0, min(1,level)) || level <= 0 { continue }
                let k = allocated*20
                points[k] = x; points[k+1] = y; points[k+2] = z; points[k+3] = 8
                allocated += 1
            }}
        }
        buffers[16].contents().bindMemory(to:UInt32.self,capacity:32)[0]=UInt32(allocated)
        initial = allocated;creamRemaining=0;creamEmitted=0
        metadata = ["in_cup": allocated, "surface_height": 0.30+0.62*level]
        surfaceDirty = true
    }
    func dispatch(_ name: String, _ count: Int, _ params: [Float], _ encoder: MTLComputeCommandEncoder) {
        if count <= 0 { return }
        let pipeline = pipelines[name]!
        encoder.setComputePipelineState(pipeline)
        for i in 0..<buffers.count where i != 10 { encoder.setBuffer(buffers[i], offset: 0, index: i) }
        encoder.setBytes(params, length: params.count*4, index: 10)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(128,pipeline.maxTotalThreadsPerThreadgroup),height:1,depth:1))
        encoder.memoryBarrier(scope: .buffers)
    }
    // Threadgroup counts come from fluidIndirect (128-wide groups over a GPU-built list).
    func dispatchIndirect(_ name: String, _ offset: Int, _ params: [Float], _ encoder: MTLComputeCommandEncoder) {
        let pipeline = pipelines[name]!
        encoder.setComputePipelineState(pipeline)
        for i in 0..<buffers.count where i != 10 { encoder.setBuffer(buffers[i], offset: 0, index: i) }
        encoder.setBytes(params, length: params.count*4, index: 10)
        encoder.dispatchThreadgroups(indirectBuffer: buffers[28], indirectBufferOffset: offset,
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.memoryBarrier(scope: .buffers)
    }
    func encode(_ frame: [String: Any], _ u: [Float], render: Bool, command: MTLCommandBuffer) {
        guard let rawGravity = floats(frame["gravity"],count:3), let acceleration = floats(frame["accel"],count:3),
              let dtNumber = frame["dt"] as? NSNumber, dtNumber.floatValue.isFinite,
              let stirNumber = frame["stir"] as? NSNumber, stirNumber.floatValue.isFinite,
              let events = frame["events"] as? [[String:Any]], events.count <= 32 else { fail("invalid fluid request") }
        let dt = max(0, min(0.1,dtNumber.floatValue))
        let norm = sqrt(rawGravity.reduce(0) {$0+$1*$1})
        guard norm.isFinite, norm > 0.1, norm < 4, acceleration.allSatisfy({abs($0)<=100}) else { fail("invalid fluid forces") }
        // One model unit is 10 cm, not one metre: gravity is 98.1 units/s².
        // The existing Python acceleration contract uses 20 units per g.
        let forceScale: Float = 98.1 / 20
        var gravity = rawGravity.map {$0/norm*98.1}
        for i in 0..<3 { gravity[i] -= max(-40,min(40,acceleration[i])) * forceScale }
        var sip = 0, poke: Float = 0, knock: Float = 0
        for event in events {
            guard let kind = event["kind"] as? String else { fail("invalid fluid event") }
            switch kind {
            case "reset": reset(level: 1)
            case "refill": pourRemaining = max(0,fullCount-Int((metadata["in_cup"] as? NSNumber)?.floatValue ?? Float(initial+emitted)))
            case "sip": sip += Int(Float(fullCount)*0.20)
            case "cream": creamRemaining=min(fullCount,creamRemaining+Int(Float(fullCount)*0.06))
            case "poke": poke += 3.99; knock += 0.266
            case "knock": knock += 9.30
            case "target":
                guard let level = event["level"] as? NSNumber, level.floatValue.isFinite else {fail("invalid target")}
                let desired = Int(max(0,min(1,level.floatValue))*Float(fullCount))
                let current = Int((metadata["in_cup"] as? NSNumber)?.floatValue ?? Float(initial+emitted))
                if event["fill"] as? Bool == true {
                    // Count queued and falling source particles before emitting more.
                    let falling = Int((metadata["airborne"] as? NSNumber)?.floatValue ?? 0)
                    pourRemaining = max(pourRemaining,max(0,desired-current-falling))
                } else {sip += max(0,current-desired)}
            default: fail("unknown fluid event")
            }
        }
        buffers[16].contents().bindMemory(to:UInt32.self,capacity:32)[11]=0
        let newStart = allocated
        let coffeeSpawn = min(pourRemaining, min(min(Fluid.capacity-initial-emitted,Fluid.slots-allocated),Int(dt*Float(fullCount)*0.22)))
        let milkStart=allocated+coffeeSpawn
        let milkSpawn=min(creamRemaining,min(min(Fluid.capacity-initial-emitted-coffeeSpawn,Fluid.slots-milkStart),Int(dt*Float(fullCount)*0.03)))
        let spawn=coffeeSpawn+milkSpawn
        allocated += spawn; emitted += spawn; pourRemaining -= coffeeSpawn
        creamRemaining -= milkSpawn;creamEmitted += milkSpawn
        buffers[16].contents().bindMemory(to:UInt32.self,capacity:32)[0]=UInt32(allocated)
        var p = [Float](repeating:0,count:32)
        p[1]=gravity[0];p[2]=gravity[1];p[3]=gravity[2];p[4]=Float(allocated)
        p[5]=time;p[6]=max(0,min(1,stirNumber.floatValue));p[7]=u[4];p[8]=Float(sip);p[9]=min(13.3,poke)
        p[10]=Float(newStart);p[11]=Float(allocated);p[13]=1.65
        p[14]=min(11.1,knock);p[15]=(frame["cup_motion"] as? Bool ?? true) ? 1:0
        p[16]=Float(milkStart);p[17]=Float(allocated);p[18]=Float(fullCount);p[20]=dt
        guard let blit=command.makeBlitCommandEncoder() else {fail("fluid clear encoder")}
        blit.fill(buffer:buffers[12],range:0..<256,value:0);blit.endEncoding()
        guard let encoder=command.makeComputeCommandEncoder() else {fail("fluid encoder")}
        dispatch("cupEvents",1,p,encoder)
        dispatch("fluidEvents",allocated,p,encoder)
        // Adaptive substeps: dt_sub = min(1/120, 0.8 DX / v_max), v_max from the
        // previous frame's particles and the cup's surface speed. The reference
        // rises instantly and decays over a few frames; impulses pre-arm it so a
        // knock's first frame is already finely stepped.
        let body = buffers[14].contents().bindMemory(to:Float.self, capacity:20)
        let cupSpeed = sqrt(body[8]*body[8]+body[9]*body[9]+body[10]*body[10])
            + 1.1*sqrt(body[12]*body[12]+body[13]*body[13]+body[14]*body[14])
        let measured = max(cupSpeed, (metadata["max_speed"] as? NSNumber)?.floatValue ?? 18)
        speedReference = max(measured, speedReference*0.8)
        if knock > 0 || poke > 0 { speedReference = max(speedReference, 18) }
        let gridDX: Float = 0.085
        let substep = min(1/120, 0.8*gridDX/max(speedReference, 1e-3))
        let steps = dt>0 ? min(Int(ceil(dt*480)), max(1,Int(ceil(dt/substep)))) : 0
        if steps>0 {
            p[0]=dt/Float(steps);lastSteps=steps
            p[23]=min(24, gridDX/p[0])
            for _ in 0..<steps {
                time += p[0];p[5]=time
                dispatch("cupStep",1,p,encoder)
                dispatch("fluidClear",Fluid.cells,p,encoder)
                dispatch("fluidBins",allocated,p,encoder)
                dispatch("fluidP2G",Fluid.cells,p,encoder)
                dispatch("fluidIndirect",1,p,encoder)
                dispatch("fluidVorticity",Fluid.cells,p,encoder)
                dispatch("fluidForces",Fluid.cells,p,encoder)
                dispatch("fluidDivergence",Fluid.cells,p,encoder)
                for _ in 0..<40 {
                    p[12]=0;dispatchIndirect("fluidPressure",16,p,encoder)
                    p[12]=1;dispatchIndirect("fluidPressure",32,p,encoder)
                }
                dispatch("fluidProject",Fluid.cells,p,encoder)
                p[22]=0;dispatch("fluidExtrapolate",Fluid.cells,p,encoder)
                p[22]=1;dispatch("fluidExtrapolate",Fluid.cells,p,encoder)
                dispatch("fluidReaction",Fluid.cells,p,encoder)
                dispatch("fluidG2P",allocated,p,encoder)
            }
            dispatch("fluidGridStats",Fluid.cells,p,encoder)
            dispatch("fluidBreakup",allocated,p,encoder)
            dispatch("filmGather",Fluid.filmCells,p,encoder)
            let filmSteps=max(1,Int(ceil(dt/(1/90))))
            p[21]=dt/Float(filmSteps)
            for _ in 0..<filmSteps {
                dispatch("filmFlux",Fluid.filmCells,p,encoder)
                dispatch("filmAdvance",Fluid.filmCells,p,encoder)
                dispatch("filmCommit",Fluid.filmCells,p,encoder)
            }
            surfaceDirty = true
        }
        if !events.isEmpty || spawn>0 {surfaceDirty=true}
        if render && surfaceDirty {
            dispatch("fluidClear",Fluid.cells,p,encoder)
            dispatch("fluidBins",Fluid.slots,p,encoder)
            dispatch("fluidSurface",Fluid.surfaceCells,p,encoder)
            surfaceDirty = false
        }
        dispatch("fluidStats",Fluid.slots,p,encoder)
        dispatch("filmStats",Fluid.filmCells,p,encoder)
        encoder.endEncoding()
    }
    func readStats() -> [String: Any] {
        let a=buffers[12].contents().bindMemory(to:UInt32.self,capacity:64)
        let details=buffers[16].contents().bindMemory(to:UInt32.self,capacity:32)
        allocated=Int(details[0])
        let names=["in_cup","on_saucer","on_table","airborne","off_scene","sipped","invalid"]
        var out:[String:Any]=[:]
        for (i,key) in names.enumerated(){out[key]=Float(a[i])/8}
        out["solver"]="apic";out["allocated"]=initial+emitted;out["slots_used"]=allocated;out["slot_capacity"]=Fluid.slots;out["initial"]=initial
        out["emitted"]=emitted;out["capacity"]=Fluid.capacity
        out["pour_remaining"]=pourRemaining;out["capacity_limited"]=(pourRemaining>0 || creamRemaining>0) && (initial+emitted==Fluid.capacity || allocated==Fluid.slots)
        out["cream_remaining"]=creamRemaining;out["cream_emitted"]=creamEmitted
        out["cream_mass"]=(Float(a[14])+Float(details[4]))/65536
        out["cream_in_cup"]=Float(a[15])/65536
        out["cream_fraction"]=Float(a[15])/65536/max(0.125,Float(a[0])/8)
        out["cream_variance"]=max(0,Float(a[16])/65536/max(0.125,Float(a[0])/8)-pow(Float(a[15])/65536/max(0.125,Float(a[0])/8),2))
        out["cup_fraction"]=Float(a[0])/8/Float(max(1,fullCount))
        out["mean_speed2"]=Float(a[7])/100/max(0.125,Float(a[48])/8)
        out["max_speed"]=sqrt(Float(a[31])/1000)
        out["max_penetration"]=Float(a[8])/1000000
        out["surface_height"]=max(0.15,min(2.2,Float(a[9])/1000000))
        out["divergence_before"]=Float(a[11])/1000/Float(max(1,a[13]))
        out["divergence_after"]=Float(a[12])/1000/Float(max(1,a[13]))
        out["simulated_time"]=time
        let b=buffers[14].contents().bindMemory(to:Float.self,capacity:20)
        out["cup_position"]=(0..<3).map {b[$0]}
        out["cup_rotation"]=(4..<8).map {b[$0]}
        out["cup_velocity"]=(8..<11).map {b[$0]}
        out["cup_angular"]=(12..<15).map {b[$0]}
        out["cup_tilt"]=acos(max(-1,min(1,1-2*(b[4]*b[4]+b[6]*b[6]))))*180 / .pi
        out["cup_impact"]=Float(a[20])/10000
        out["cup_penetration"]=Float(a[21])/1000000
        out["splash_impact"]=Float(a[22])/100
        out["body_invalid"]=Int(a[24])
        out["mean_cup_height"]=Float(a[26])/1000/max(0.125,Float(a[0])/8)
        out["max_grid_density"]=Float(a[27])/1000
        out["occupied_cells"]=Int(a[28]);out["compressed_cells"]=Int(a[29])
        out["grid_volume"]=Float(a[30])/1000000
        let deposited=Int(details[2])+Int(details[3])
        out["on_saucer"]=(Float(a[1])+Float(details[2]))/8
        out["on_table"]=(Float(a[2])+Float(details[3]))/8
        out["split_events"]=Int(details[1]);out["rim_drips"]=Int(details[8])
        out["spray_droplets"]=Int(a[40]);out["spray_volume"]=Float(a[41])/8
        out["surface_impacts"]=Int(details[6]);out["cup_ripples"]=Int(details[7])
        out["deposited_volume"]=Float(deposited)/8
        out["film_quanta"]=Int(a[32]);out["evaporated_quanta"]=Int(a[33])
        out["film_volume"]=Float(a[32])/(8*4096);out["evaporated_volume"]=Float(a[33])/(8*4096)
        out["film_balance_error"]=Int(a[32])+Int(a[33])-deposited*4096
        out["wet_cells"]=Int(a[36]);out["stained_cells"]=Int(a[37])
        out["coffee_residue"]=Float(a[38])/(8*4096);out["cream_residue"]=Float(a[39])/(8*4096)
        out["split_momentum_error"]=Float(details[9])/1e6
        out["spray_max_radius"]=Float(a[43])/1e6
        out["wet_footprint_cells"]=Int(a[44]);out["edge_residue"]=Float(a[45])/(8*4096)
        let particleVolume:Float=0.0425*0.0425*0.0425
        let filmCellWidth:Float=8.0/384
        let filmQuantumVolume:Float=particleVolume/(8*4096)
        out["max_film_depth"]=Float(a[46])*filmQuantumVolume/(filmCellWidth*filmCellWidth)
        out["ripple_height"]=Float(a[42])/1e8
        out["substeps"]=lastSteps
        metadata=out
        return out
    }
}
FileHandle.standardOutput.write(Data("CAF_METAL_5\n".utf8))
var output: MTLBuffer?, outputSize=0
var fluid: Fluid?
guard let restingBody=device.makeBuffer(length:80,options:.storageModeShared) else {fail("body buffer")}
memset(restingBody.contents(),0,80)
let resting=restingBody.contents().bindMemory(to:Float.self,capacity:20)
resting[1]=0.52;resting[7]=1
while let line=readLine() {
    autoreleasepool {
        guard line.utf8.count<1_000_000, let data=line.data(using:.utf8),
              let frame=(try? JSONSerialization.jsonObject(with:data)) as? [String:Any],
              var u=floats(frame["u"],count:36),let water=floats(frame["water"],count:54*26),
              let dots=floats(frame["dots"]),dots.count%4==0,dots.count<=420*4,
              u[0]>=16,u[0]<=1280,u[1]>=16,u[1]<=960,
              u[0].rounded()==u[0],u[1].rounded()==u[1],u[29]>=0,u[29]<=420,
              u[29].rounded()==u[29],Int(u[29])==dots.count/4 else {fail("invalid frame")}
        let width=Int(u[0]),height=Int(u[1]),size=width*height*3
        let shouldRender=frame["render"] as? Bool ?? true
        let enabled=u[32]>0.5
        if enabled && fluid==nil {fluid=Fluid(level:u[3],device:device)}
        if enabled {u[33]=Float((fluid!.metadata["surface_height"] as? NSNumber)?.floatValue ?? 0.9)}
        if size != outputSize {output=device.makeBuffer(length:size,options:.storageModeShared);outputSize=size}
        guard let command=queue.makeCommandBuffer() else {fail("command allocation")}
        if enabled {
            guard let settings=frame["fluid"] as? [String:Any] else {fail("missing fluid settings")}
            fluid!.encode(settings,u,render:shouldRender,command:command)
        }
        if enabled {u[35]=fluid!.time}
        if shouldRender {
            guard let target=output,
                  let uniforms=device.makeBuffer(bytes:u,length:u.count*4,options:.storageModeShared),
                  let waves=device.makeBuffer(bytes:water,length:water.count*4,options:.storageModeShared),
                  let particles=device.makeBuffer(bytes:dots.isEmpty ? [Float](repeating:0,count:4):dots,
                                                   length:max(4,dots.count)*4,options:.storageModeShared),
                  let encoder=command.makeComputeCommandEncoder() else {fail("render allocation")}
            let pipeline=pipelines["coffee"]!
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(target,offset:0,index:0);encoder.setBuffer(uniforms,offset:0,index:1)
            encoder.setBuffer(enabled ? fluid!.buffers[11]:waves,offset:0,index:2)
            encoder.setBuffer(particles,offset:0,index:3)
            encoder.setBuffer(enabled ? fluid!.buffers[14]:restingBody,offset:0,index:4)
            // Legacy shaders do not read these, but every pointer has a valid binding.
            for (index,source) in [(5,0),(6,1),(7,2),(8,3),(9,18),(10,20),(11,22),(12,24)] {
                encoder.setBuffer(enabled ? fluid!.buffers[source]:restingBody,offset:0,index:index)
            }
            let x=pipeline.threadExecutionWidth,y=min(8,pipeline.maxTotalThreadsPerThreadgroup/x)
            encoder.dispatchThreads(MTLSize(width:width,height:height,depth:1),threadsPerThreadgroup:MTLSize(width:x,height:y,depth:1))
            encoder.endEncoding()
        }
        command.commit();command.waitUntilCompleted()
        guard command.status == .completed else {fail(command.error.map(String.init(describing:)) ?? "GPU failed")}
        var metadata:[String:Any]=enabled ? fluid!.readStats():["solver":"legacy"]
        metadata["gpu_ms"]=(command.gpuEndTime-command.gpuStartTime)*1000
        guard let encoded=try? JSONSerialization.data(withJSONObject:metadata,options:[.sortedKeys]) else {fail("invalid fluid diagnostics")}
        writePacket(encoded)
        writePacket(shouldRender ? Data(bytes:output!.contents(),count:size):Data())
    }
}
