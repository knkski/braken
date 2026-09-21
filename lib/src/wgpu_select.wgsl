const BRANCH_OPEN: u32 = 0xffffffffu;
const BRANCH_CLOSE: u32 = 0xfffffffeu;
const WILDCARD_RULE: u32 = 0xfffffffdu;
const NO_RULE: u32 = 0xffffffffu;

struct Params {
    input_len: u32,
    global_base_lo: u32,
    global_base_hi: u32,
    rule_count: u32,
    seed_lo: u32,
    seed_hi: u32,
    iteration_lo: u32,
    iteration_hi: u32,
    ambiguous_policy: u32,
    padding_0: u32,
    padding_1: u32,
    padding_2: u32,
};

struct Rule {
    lhs: u32,
    weight_bits: u32,
    rhs_offset: u32,
    rhs_len: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read> rules: array<Rule>;
@group(0) @binding(3) var<storage, read_write> decisions: array<u32>;
@group(0) @binding(4) var<storage, read_write> lengths: array<u32>;

fn hash_word(value: u32) -> u32 {
    var x = value;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}

// The key is based only on the public seed, generation, and absolute symbol
// position. Dispatch and buffer chunking therefore cannot change a choice.
fn random_unit(position_lo: u32, position_hi: u32) -> f32 {
    var key = params.seed_lo ^ hash_word(params.seed_hi + 0x9e3779b9u);
    key = key ^ hash_word(params.iteration_lo + 0x85ebca6bu);
    key = key ^ hash_word(params.iteration_hi + 0xc2b2ae35u);
    key = key ^ hash_word(position_lo + 0x27d4eb2fu);
    // hash_word(0) is zero, preserving seeded results below 2^32 while still
    // distinguishing every non-zero high half.
    key = key ^ hash_word(position_hi);
    let random_bits = hash_word(key) >> 8u;
    return f32(random_bits) * (1.0 / 16777216.0);
}

@compute @workgroup_size(256)
fn select_rules_and_lengths(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let local_index = global_id.x;
    if local_index >= params.input_len {
        return;
    }

    let token = input[local_index];
    var selected = NO_RULE;
    var output_len = 1u;

    if token != BRANCH_OPEN && token != BRANCH_CLOSE {
        var total_weight = 0.0;
        var match_count = 0u;
        var first = NO_RULE;
        var last = NO_RULE;
        var rule_index = 0u;
        loop {
            if rule_index >= params.rule_count {
                break;
            }
            let rule = rules[rule_index];
            if rule.lhs == token || rule.lhs == WILDCARD_RULE {
                if first == NO_RULE {
                    first = rule_index;
                }
                last = rule_index;
                total_weight += bitcast<f32>(rule.weight_bits);
                match_count += 1u;
            }
            rule_index += 1u;
        }

        selected = first;
        let position_lo = params.global_base_lo + local_index;
        let position_carry = select(0u, 1u, position_lo < params.global_base_lo);
        let position_hi = params.global_base_hi + position_carry;
        let random = random_unit(position_lo, position_hi);
        if first != NO_RULE && total_weight > 0.0 {
            selected = last;
            var sample = random * total_weight;
            rule_index = 0u;
            loop {
                if rule_index >= params.rule_count {
                    break;
                }
                let rule = rules[rule_index];
                if rule.lhs == token || rule.lhs == WILDCARD_RULE {
                    sample -= bitcast<f32>(rule.weight_bits);
                    if sample < 0.0 {
                        selected = rule_index;
                        break;
                    }
                }
                rule_index += 1u;
            }
        } else if first != NO_RULE && match_count > 1u && params.ambiguous_policy == 0u {
            let choice_target = u32(random * f32(match_count));
            var seen = 0u;
            rule_index = 0u;
            loop {
                if rule_index >= params.rule_count {
                    break;
                }
                let rule = rules[rule_index];
                if rule.lhs == token || rule.lhs == WILDCARD_RULE {
                    if seen == choice_target {
                        selected = rule_index;
                        break;
                    }
                    seen += 1u;
                }
                rule_index += 1u;
            }
        }

        if selected != NO_RULE {
            output_len = rules[selected].rhs_len;
        }
    }

    decisions[local_index] = selected;
    lengths[local_index] = output_len;
}
