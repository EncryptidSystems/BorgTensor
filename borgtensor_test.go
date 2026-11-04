package main

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func writeF32(path string, tensors []struct {
	name  string
	shape []int
	v     []float32
}) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	w, err := NewSTW(path, nil)
	if err != nil {
		return err
	}
	for _, t := range tensors {
		w.AddF32(t.name, t.shape, t.v)
	}
	return w.Close(nil)
}

func fill(n int, v float32) []float32 {
	a := make([]float32, n)
	for i := range a {
		a[i] = v
	}
	return a
}

func nearConst(t *testing.T, a []float32, v float32) {
	t.Helper()
	for i, x := range a {
		if math.Abs(float64(x-v)) > 1e-3 {
			t.Fatalf("value %d = %v, want %v", i, x, v)
		}
	}
}

func TestShapeJointIsSpline(t *testing.T) {
	const R, C, Rt, Ct = 5, 7, 4, 6
	rng := rand.New(rand.NewSource(3))
	src := make([]float32, R*C)
	for i := range src {
		src[i] = float32(rng.NormFloat64())
	}
	got := shapeJoint(src, R, C, Rt, Ct)
	G := [2]int{gridSize(R, 0), gridSize(C, 0)}
	grid := splineGrid(src, R, C, linCoords(R), linCoords(C), G, 0)
	want := sampleGrid(grid, G, linCoords(Rt), linCoords(Ct))
	if len(got) != len(want) {
		t.Fatalf("len %d vs %d", len(got), len(want))
	}
	var mx float64
	for i := range got {
		mx = math.Max(mx, math.Abs(float64(got[i])-want[i]))
	}
	if mx > 1e-5 {
		t.Fatalf("max |joint - splineGrid/sampleGrid| = %g", mx)
	}
	nearConst(t, shapeJoint(fill(R*C, 1), R, C, Rt, Ct), 1)
}

func TestAxisMatchesJoint(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	src := make([]float32, 8)
	for i := range src {
		src[i] = float32(rng.NormFloat64())
	}
	got := resampleAxis(src, []int{8}, 0, 5)
	want := shapeJoint(src, 8, 1, 5, 1)
	if len(got) != len(want) {
		t.Fatalf("len %d vs %d", len(got), len(want))
	}
	var mx float64
	for i := range got {
		mx = math.Max(mx, math.Abs(float64(got[i])-float64(want[i])))
	}
	if mx > 1e-5 {
		t.Fatalf("max |axis - joint| = %g", mx)
	}
	// Expert rows stay apart when only the hidden axis changes.
	row := make([]float32, 2*8)
	for i := 0; i < 8; i++ {
		row[i] = 1
		row[8+i] = -1
	}
	out, err := shapeND(row, []int{2, 8}, []int{2, 5})
	if err != nil {
		t.Fatal(err)
	}
	nearConst(t, out[:5], 1)
	nearConst(t, out[5:], -1)
}

func TestTargetShapeSquareHeads(t *testing.T) {
	src := geom{D: 8, Q: 8, KV: 4, F: 12, Vocab: 6, Layers: 2}
	tgt := geom{D: 4, Q: 6, KV: 2, F: 10, Vocab: 9, Layers: 3}
	q, err := targetShape("model.layers.0.self_attn.q_proj.weight", []int{8, 8}, src, tgt)
	if err != nil {
		t.Fatal(err)
	}
	if !sameInts(q, []int{6, 4}) {
		t.Fatalf("q %v", q)
	}
	k, err := targetShape("model.layers.0.self_attn.k_proj.weight", []int{8, 4}, src, tgt)
	if err != nil {
		t.Fatal(err)
	}
	if !sameInts(k, []int{4, 2}) {
		t.Fatalf("transposed k %v", k)
	}
}

func TestBorgLlamaShards(t *testing.T) {
	dir := t.TempDir()
	s0 := filepath.Join(dir, "src", "model-00001-of-00002.safetensors")
	s1 := filepath.Join(dir, "src", "model-00002-of-00002.safetensors")
	model := filepath.Join(dir, "model", "model.safetensors")
	out := filepath.Join(dir, "out")
	layer := func(n int, v float32) []struct {
		name  string
		shape []int
		v     []float32
	} {
		p := "model.layers." + itoa(n) + "."
		return []struct {
			name  string
			shape []int
			v     []float32
		}{
			{p + "input_layernorm.weight", []int{8}, fill(8, v)},
			{p + "self_attn.q_proj.weight", []int{8, 8}, fill(64, v)},
			{p + "self_attn.k_proj.weight", []int{4, 8}, fill(32, v)},
			{p + "mlp.gate_proj.weight", []int{12, 8}, fill(96, v)},
			{p + "mlp.down_proj.weight", []int{8, 12}, fill(96, v)},
		}
	}
	var a []struct {
		name  string
		shape []int
		v     []float32
	}
	a = append(a, struct {
		name  string
		shape []int
		v     []float32
	}{"model.embed_tokens.weight", []int{6, 8}, fill(48, 1)})
	a = append(a, layer(0, 1)...)
	if err := writeF32(s0, a); err != nil {
		t.Fatal(err)
	}
	var b []struct {
		name  string
		shape []int
		v     []float32
	}
	b = append(b, layer(1, -1)...)
	b = append(b, struct {
		name  string
		shape []int
		v     []float32
	}{"model.norm.weight", []int{8}, fill(8, 1)})
	if err := writeF32(s1, b); err != nil {
		t.Fatal(err)
	}
	tgt := []struct {
		name  string
		shape []int
		v     []float32
	}{
		{"model.embed_tokens.weight", []int{9, 6}, fill(54, 0)},
		{"model.norm.weight", []int{6}, fill(6, 0)},
		{"model.layers.0.self_attn.k_proj.weight", []int{2, 6}, fill(12, 0)},
		{"model.layers.0.mlp.gate_proj.weight", []int{10, 6}, fill(60, 0)},
		{"model.layers.2.self_attn.q_proj.weight", []int{6, 6}, fill(36, 0)},
	}
	if err := writeF32(model, tgt); err != nil {
		t.Fatal(err)
	}
	if err := borgTensor([]string{"-src", s0, "-src", s1, "-model", model, "-out", out}); err != nil {
		t.Fatal(err)
	}
	idxb, err := os.ReadFile(filepath.Join(out, "model.safetensors.index.json"))
	if err != nil {
		t.Fatal(err)
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(idxb, &idx); err != nil {
		t.Fatal(err)
	}
	if idx.WeightMap["model.embed_tokens.weight"] != "model-00001-of-00002.safetensors" {
		t.Fatalf("embed shard %s", idx.WeightMap["model.embed_tokens.weight"])
	}
	if idx.WeightMap["model.norm.weight"] != "model-00002-of-00002.safetensors" {
		t.Fatalf("norm shard %s", idx.WeightMap["model.norm.weight"])
	}
	if idx.WeightMap["model.layers.0.self_attn.q_proj.weight"] != "model-00001-of-00002.safetensors" {
		t.Fatalf("layer 0 shard %s", idx.WeightMap["model.layers.0.self_attn.q_proj.weight"])
	}
	if idx.WeightMap["model.layers.2.self_attn.q_proj.weight"] != "model-00002-of-00002.safetensors" {
		t.Fatalf("layer 2 shard %s", idx.WeightMap["model.layers.2.self_attn.q_proj.weight"])
	}
	st0, err := openST(filepath.Join(out, "model-00001-of-00002.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer st0.Close()
	st1, err := openST(filepath.Join(out, "model-00002-of-00002.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer st1.Close()
	check := func(st *ST, name string, shape []int, v float32) {
		t.Helper()
		if !sameInts(st.H[name].Shape, shape) {
			t.Fatalf("%s shape %v", name, st.H[name].Shape)
		}
		if st.H[name].Dtype != "BF16" {
			t.Fatalf("%s dtype %s", name, st.H[name].Dtype)
		}
		a, err := st.F32(name)
		if err != nil {
			t.Fatal(err)
		}
		nearConst(t, a, v)
	}
	check(st0, "model.embed_tokens.weight", []int{9, 6}, 1)
	check(st0, "model.layers.0.self_attn.q_proj.weight", []int{6, 6}, 1)
	check(st0, "model.layers.0.self_attn.k_proj.weight", []int{2, 6}, 1)
	check(st0, "model.layers.0.mlp.gate_proj.weight", []int{10, 6}, 1)
	check(st0, "model.layers.0.mlp.down_proj.weight", []int{6, 10}, 1)
	check(st0, "model.layers.0.input_layernorm.weight", []int{6}, 1)
	check(st1, "model.layers.1.self_attn.q_proj.weight", []int{6, 6}, -1)
	check(st1, "model.layers.2.self_attn.q_proj.weight", []int{6, 6}, -1)
	check(st1, "model.norm.weight", []int{6}, 1)
}

func TestBorgExperts(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "src.safetensors")
	model := filepath.Join(dir, "model.safetensors")
	out := filepath.Join(dir, "out.safetensors")
	gu := fill(2*8*24, 0)
	for i := 0; i < 8*24; i++ {
		gu[i] = 1
		gu[8*24+i] = -1
	}
	if err := writeF32(src, []struct {
		name  string
		shape []int
		v     []float32
	}{
		{"model.layers.0.self_attn.q_proj.weight", []int{8, 8}, fill(64, 1)},
		{"model.layers.0.self_attn.k_proj.weight", []int{4, 8}, fill(32, 1)},
		{"model.layers.0.mlp.experts.gate_up_proj", []int{2, 8, 24}, gu},
		{"model.layers.0.mlp.experts.down_proj", []int{2, 12, 8}, fill(2*12*8, 1)},
		{"model.layers.0.mlp.router.weight", []int{2, 8}, append(fill(8, 1), fill(8, -1)...)},
	}); err != nil {
		t.Fatal(err)
	}
	if err := writeF32(model, []struct {
		name  string
		shape []int
		v     []float32
	}{
		{"model.layers.0.self_attn.q_proj.weight", []int{6, 6}, fill(36, 0)},
		{"model.layers.0.self_attn.k_proj.weight", []int{2, 6}, fill(12, 0)},
		{"model.layers.0.mlp.experts.gate_up_proj", []int{2, 6, 20}, fill(2*6*20, 0)},
		{"model.layers.0.mlp.experts.down_proj", []int{2, 10, 6}, fill(2*10*6, 0)},
		{"model.layers.0.mlp.router.weight", []int{2, 6}, fill(12, 0)},
	}); err != nil {
		t.Fatal(err)
	}
	if err := borgTensor([]string{"-src", src, "-model", model, "-out", out}); err != nil {
		t.Fatal(err)
	}
	st, err := openST(out)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	guOut, err := st.F32("model.layers.0.mlp.experts.gate_up_proj")
	if err != nil {
		t.Fatal(err)
	}
	if !sameInts(st.H["model.layers.0.mlp.experts.gate_up_proj"].Shape, []int{2, 6, 20}) {
		t.Fatalf("gate_up %v", st.H["model.layers.0.mlp.experts.gate_up_proj"].Shape)
	}
	nearConst(t, guOut[:6*20], 1)
	nearConst(t, guOut[6*20:], -1)
	rt, err := st.F32("model.layers.0.mlp.router.weight")
	if err != nil {
		t.Fatal(err)
	}
	if !sameInts(st.H["model.layers.0.mlp.router.weight"].Shape, []int{2, 6}) {
		t.Fatalf("router %v", st.H["model.layers.0.mlp.router.weight"].Shape)
	}
	nearConst(t, rt[:6], 1)
	nearConst(t, rt[6:], -1)
	if !sameInts(st.H["model.layers.0.mlp.experts.down_proj"].Shape, []int{2, 10, 6}) {
		t.Fatalf("down %v", st.H["model.layers.0.mlp.experts.down_proj"].Shape)
	}
}

func TestBorgGrokNames(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "src.safetensors")
	model := filepath.Join(dir, "model.safetensors")
	out := filepath.Join(dir, "out.safetensors")
	q := "decoder_layer_0/multi_head_attention/query_dense/kernel"
	k := "decoder_layer_0/multi_head_attention/key_dense/kernel"
	r := "decoder_layer_0/moe/router/kernel"
	e := "embedding_layer/embedding"
	if err := writeF32(src, []struct {
		name  string
		shape []int
		v     []float32
	}{
		{q, []int{8, 8}, fill(64, 1)},
		{k, []int{4, 8}, fill(32, 1)},
		{r, []int{2, 8}, append(fill(8, 1), fill(8, -1)...)},
		{e, []int{6, 8}, fill(48, 1)},
	}); err != nil {
		t.Fatal(err)
	}
	qt := "decoder_layer_0/multi_head_attention/query_dense/kernel"
	if err := writeF32(model, []struct {
		name  string
		shape []int
		v     []float32
	}{
		{qt, []int{6, 6}, fill(36, 0)},
		{"decoder_layer_0/multi_head_attention/key_dense/kernel", []int{2, 6}, fill(12, 0)},
		{"decoder_layer_0/moe/router/kernel", []int{2, 6}, fill(12, 0)},
		{"embedding_layer/embedding", []int{9, 6}, fill(54, 0)},
	}); err != nil {
		t.Fatal(err)
	}
	if err := borgTensor([]string{"-src", src, "-model", model, "-out", out}); err != nil {
		t.Fatal(err)
	}
	st, err := openST(out)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	if !sameInts(st.H[q].Shape, []int{6, 6}) {
		t.Fatalf("query %v", st.H[q].Shape)
	}
	if !sameInts(st.H[e].Shape, []int{9, 6}) {
		t.Fatalf("embedding %v", st.H[e].Shape)
	}
	rt, err := st.F32(r)
	if err != nil {
		t.Fatal(err)
	}
	nearConst(t, rt[:6], 1)
	nearConst(t, rt[6:], -1)
}
