const BLOCK_SIZE: u32 = 256u;

struct Params {
    input_len: u32,
    block_base: u32,
    _padding_0: u32,
    _padding_1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(4) var<storage, read_write> overflow: atomic<u32>;

var<workgroup> scratch: array<u32, 256>;

fn checked_add(left: u32, right: u32) -> u32 {
    let result = left + right;
    if result < left {
        atomicStore(&overflow, 1u);
    }
    return result;
}

@compute @workgroup_size(256)
fn scan_blocks(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let lane = local_id.x;
    let index = workgroup_id.x * BLOCK_SIZE + lane;
    var value = 0u;
    if index < params.input_len {
        value = input[index];
    }
    scratch[lane] = value;
    workgroupBarrier();

    var stride = 1u;
    loop {
        if stride >= BLOCK_SIZE {
            break;
        }
        let scratch_index = (lane + 1u) * stride * 2u - 1u;
        if scratch_index < BLOCK_SIZE {
            scratch[scratch_index] = checked_add(scratch[scratch_index], scratch[scratch_index - stride]);
        }
        workgroupBarrier();
        stride *= 2u;
    }

    if lane == 0u {
        block_sums[params.block_base + workgroup_id.x] = scratch[BLOCK_SIZE - 1u];
        scratch[BLOCK_SIZE - 1u] = 0u;
    }
    workgroupBarrier();

    stride = BLOCK_SIZE / 2u;
    loop {
        let scratch_index = (lane + 1u) * stride * 2u - 1u;
        if scratch_index < BLOCK_SIZE {
            let left = scratch[scratch_index - stride];
            let right = scratch[scratch_index];
            scratch[scratch_index - stride] = right;
            scratch[scratch_index] = checked_add(right, left);
        }
        workgroupBarrier();
        if stride == 1u {
            break;
        }
        stride /= 2u;
    }

    if index < params.input_len {
        offsets[index] = scratch[lane];
    }
}
