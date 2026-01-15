
@binding(0)
@group(0)
var<storage, read_write> data: array<f32>;

@compute
@workgroup_size(1,1,1)
fn computeSomething(
    @builtin(global_invocation_id) id: vec3u
) {
    let i = id.x;
    data[i] = 2.0 * data[i];
}
