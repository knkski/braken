const NO_RULE: u32 = 0xffffffffu;

struct Params {
    input_len: u32,
    output_binding_base: u32,
    _padding_0: u32,
    _padding_1: u32,
};

struct Rule {
    lhs: u32,
    weight_bits: u32,
    rhs_offset: u32,
    rhs_len: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read> decisions: array<u32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read> rules: array<Rule>;
@group(0) @binding(5) var<storage, read> rhs: array<u32>;
@group(0) @binding(6) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(256)
fn rewrite_tokens(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= params.input_len {
        return;
    }

    let output_offset = offsets[index] - params.output_binding_base;
    let selected = decisions[index];
    if selected == NO_RULE {
        if output_offset < arrayLength(&output) {
            output[output_offset] = input[index];
        }
        return;
    }

    let rule = rules[selected];
    var rhs_index = 0u;
    loop {
        if rhs_index >= rule.rhs_len {
            break;
        }
        let destination = output_offset + rhs_index;
        if destination < arrayLength(&output) {
            output[destination] = rhs[rule.rhs_offset + rhs_index];
        }
        rhs_index += 1u;
    }
}
