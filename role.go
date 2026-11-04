package main

import "regexp"

// qkv is tested first: a fused query_key_value must not be taken for q. GPT-2's c_attn/c_fc/c_proj are
// not recognized: they are Conv1D weights stored [in, out] (transposed against every other role here),
// and attn.c_proj is the attention output, not down. A bare "gate" is not gate_proj: in MoE models
// (mlp.gate, block_sparse_moe.gate) it is the router.
var roleOrder = []string{"qkv", "q", "k", "v", "o", "gate", "up", "down", "embed", "lm_head"}
var rolePat = map[string]*regexp.Regexp{
	"q": regexp.MustCompile(`(q_proj|\bwq\b|\bquery\b|\.q\b)`), "k": regexp.MustCompile(`(k_proj|\bwk\b|\bkey\b|\.k\b)`),
	"v": regexp.MustCompile(`(v_proj|\bwv\b|\bvalue\b|\.v\b)`), "o": regexp.MustCompile(`(o_proj|\bwo\b|out_proj|attn\.dense|attention\.output\.dense)`),
	"gate": regexp.MustCompile(`(gate_proj|\bw1\b)`), "up": regexp.MustCompile(`(up_proj|\bw3\b|\bfc1\b|dense_h_to_4h)`),
	"down": regexp.MustCompile(`(down_proj|\bw2\b|\bfc2\b|dense_4h_to_h)`), "qkv": regexp.MustCompile(`(qkv_proj|\bqkv\b|\bWqkv\b|query_key_value)`),
	"embed": regexp.MustCompile(`(embed_tokens|wte|word_embeddings|tok_embeddings)`), "lm_head": regexp.MustCompile(`(lm_head|output\.weight$)`),
}

// Paper: no equation (helper: I/O, progress or utility)
func roleOf(n string) string {
	for _, r := range roleOrder {
		if rolePat[r].MatchString(n) {
			return r
		}
	}
	return ""
}
