const BLOCK_SIZE: u32 = 256u;

struct Params {
    input_len: u32,
    block_base: u32,
    _padding_0: u32,
    _padding_1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(2) var<storage, read> block_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> overflow: atomic<u32>;

@compute @workgroup_size(256)
fn add_block_offsets(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= params.input_len {
        return;
    }
    let left = offsets[index];
    let right = block_offsets[params.block_base + index / BLOCK_SIZE];
    let result = left + right;
    if result < left {
        atomicStore(&overflow, 1u);
    }
    offsets[index] = result;
}
