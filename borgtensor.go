package main

// BorgTensor shapes any safetensors checkpoint onto a model's geometry.
//
// The path is the one used for Llama 3.2: tensor, then the source shard that
// holds it, then the output shard with that same index, then the shaped tensor.
// A matrix is resampled by splineGrid and sampleGrid (linCoords on each axis).
// A vector, or one axis of a wider tensor, is that same path with a single column.
// The mixer is not involved. Tensor ids stay. Layer count and axis lengths become
// the target model's.

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

type borgPaths []string

func (p *borgPaths) String() string { return strings.Join(*p, ",") }

func (p *borgPaths) Set(s string) error {
	*p = append(*p, s)
	return nil
}

func borgTensor(args []string) error {
	fs := flag.NewFlagSet("BorgTensor", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var src borgPaths
	var model, out string
	fs.Var(&src, "src", "source safetensors file or directory (repeatable)")
	fs.StringVar(&model, "model", "", "target model safetensors file or directory")
	fs.StringVar(&out, "out", "", "output file or directory")
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, `BorgTensor shapes any safetensors checkpoint onto a model's geometry.

  tensor → source shard → output shard → shaped tensor

  borgtensor -src SRC -model MODEL -out OUT

`)
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return nil
		}
		return err
	}
	if len(src) == 0 || model == "" || out == "" {
		fs.Usage()
		return fmt.Errorf("-src, -model, and -out are required")
	}
	srcFiles, err := listShards(src)
	if err != nil {
		return err
	}
	modelFiles, err := listShards([]string{model})
	if err != nil {
		return err
	}
	source, err := openCatalog(srcFiles)
	if err != nil {
		return err
	}
	defer source.close()
	target, err := openCatalog(modelFiles)
	if err != nil {
		return err
	}
	defer target.close()
	sg, err := readGeom(source.items)
	if err != nil {
		return fmt.Errorf("source: %w", err)
	}
	tg, err := readGeom(target.items)
	if err != nil {
		return fmt.Errorf("model: %w", err)
	}
	if sg.D == 0 || tg.D == 0 {
		return fmt.Errorf("source D=%d model D=%d: a checkpoint needs a residual width", sg.D, tg.D)
	}
	logf("BorgTensor source D=%d F=%d Q=%d KV=%d vocab=%d experts=%d layers=%d", sg.D, sg.F, sg.Q, sg.KV, sg.Vocab, sg.Experts, sg.Layers)
	logf("BorgTensor model  D=%d F=%d Q=%d KV=%d vocab=%d experts=%d layers=%d", tg.D, tg.F, tg.Q, tg.KV, tg.Vocab, tg.Experts, tg.Layers)
	jobs, err := planBorg(sg, tg, source)
	if err != nil {
		return err
	}
	return writeBorg(out, source, jobs)
}

// ---- shards ----

func listShards(paths []string) ([]string, error) {
	var out []string
	for _, p := range paths {
		fi, err := os.Stat(p)
		if err != nil {
			return nil, err
		}
		if !fi.IsDir() {
			out = append(out, p)
			continue
		}
		one, err := dirShards(p)
		if err != nil {
			return nil, err
		}
		out = append(out, one...)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no safetensors")
	}
	return out, nil
}

func dirShards(dir string) ([]string, error) {
	idx := filepath.Join(dir, "model.safetensors.index.json")
	if b, err := os.ReadFile(idx); err == nil {
		var h struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if json.Unmarshal(b, &h) == nil && len(h.WeightMap) > 0 {
			seen := map[string]bool{}
			var names []string
			for _, f := range h.WeightMap {
				if !seen[f] {
					seen[f] = true
					names = append(names, f)
				}
			}
			sort.Strings(names)
			out := make([]string, len(names))
			for i, f := range names {
				out[i] = filepath.Join(dir, filepath.FromSlash(f))
			}
			return out, nil
		}
	}
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var sharded, all []string
	var single string
	for _, e := range ents {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".safetensors") {
			continue
		}
		p := filepath.Join(dir, e.Name())
		all = append(all, p)
		if e.Name() == "model.safetensors" {
			single = p
		}
		if strings.Contains(e.Name(), "-of-") {
			sharded = append(sharded, p)
		}
	}
	sort.Strings(sharded)
	sort.Strings(all)
	switch {
	case len(sharded) > 0:
		return sharded, nil
	case single != "":
		return []string{single}, nil
	case len(all) > 0:
		return all, nil
	default:
		return nil, fmt.Errorf("%s: no safetensors", dir)
	}
}

type srcItem struct {
	name  string
	shape []int
	shard int
}

type catalog struct {
	paths  []string
	shards []*ST
	items  []srcItem
	shard  map[string]int
}

func openCatalog(files []string) (*catalog, error) {
	c := &catalog{paths: files, shard: map[string]int{}}
	for i, p := range files {
		st, err := openST(p)
		if err != nil {
			c.close()
			return nil, err
		}
		c.shards = append(c.shards, st)
		for _, k := range st.Keys {
			if skipKey(st, k) {
				continue
			}
			if _, dup := c.shard[k]; dup {
				c.close()
				return nil, fmt.Errorf("tensor %s is in two shards", k)
			}
			sh, id := storedShape(st, k)
			if id != k {
				if _, dup := c.shard[id]; dup {
					continue // the float tensor is already the source
				}
			}
			c.shard[id] = i
			c.items = append(c.items, srcItem{id, sh, i})
		}
	}
	if len(c.items) == 0 {
		c.close()
		return nil, fmt.Errorf("no tensors")
	}
	return c, nil
}

func (c *catalog) close() {
	for _, s := range c.shards {
		_ = s.Close()
	}
}

func skipKey(st *ST, name string) bool {
	if strings.HasPrefix(name, "bnzmix.") || isLimbTensor(name) {
		return true
	}
	if strings.HasSuffix(name, "weight_scale") && st.Has(strings.TrimSuffix(name, "_scale")) {
		return true
	}
	if strings.HasSuffix(name, "_scales") && st.Has(strings.TrimSuffix(name, "_scales")+"_blocks") {
		return true
	}
	if strings.HasSuffix(name, "_blocks") && st.Has(strings.TrimSuffix(name, "_blocks")) {
		return true
	}
	return false
}

// storedShape is the float layout BorgTensor resamples. MXFP4 blocks are the
// dequantized matrix. Packed ternary is the logical row count.
func storedShape(st *ST, name string) ([]int, string) {
	if strings.HasSuffix(name, "_blocks") {
		sc := strings.TrimSuffix(name, "_blocks") + "_scales"
		bsh := st.H[name].Shape
		if st.Has(sc) && len(bsh) >= 3 {
			groups, B := bsh[len(bsh)-2], bsh[len(bsh)-1]
			row := bsh[len(bsh)-3]
			osh := append([]int{}, bsh[:len(bsh)-3]...)
			osh = append(osh, groups*B*2, row)
			return osh, strings.TrimSuffix(name, "_blocks")
		}
	}
	return append([]int(nil), st.LogicalShape(name)...), name
}

func (c *catalog) load(name string) ([]float32, []int, error) {
	i, ok := c.shard[name]
	if !ok {
		return nil, nil, fmt.Errorf("missing %s", name)
	}
	st := c.shards[i]
	if st.Has(name) {
		return st.LoadDense(name)
	}
	blocks := name + "_blocks"
	sc := name + "_scales"
	if st.Has(blocks) && st.Has(sc) && st.H[blocks].Dtype == "U8" {
		b, err := st.Raw(blocks)
		if err != nil {
			return nil, nil, err
		}
		s, err := st.Raw(sc)
		if err != nil {
			return nil, nil, err
		}
		return DequantMXFP4(b, s, st.H[blocks].Shape, st.H[sc].Shape)
	}
	return nil, nil, fmt.Errorf("missing %s", name)
}

// ---- geometry ----

type geom struct {
	D, F, Q, KV, Vocab, Experts, HD, Layers int
}

func (g *geom) note(dst *int, v int, what string) error {
	if v <= 0 {
		return nil
	}
	if *dst == 0 {
		*dst = v
		return nil
	}
	if *dst != v {
		return fmt.Errorf("%s is %d and %d", what, *dst, v)
	}
	return nil
}

func readGeom(items []srcItem) (geom, error) {
	var g geom
	for _, it := range items {
		if l := layerNum(it.name); l+1 > g.Layers {
			g.Layers = l + 1
		}
		sh := it.shape
		if len(sh) == 1 && (strings.Contains(it.name, "q_norm") || strings.Contains(it.name, "k_norm")) {
			if err := g.note(&g.HD, sh[0], "head dim"); err != nil {
				return g, err
			}
		}
	}
	// Query, gate, and the embedding name the residual. Their columns are hidden.
	for _, it := range items {
		if len(it.shape) != 2 {
			continue
		}
		sh := it.shape
		switch assimilRole(it.name) {
		case "q":
			if err := g.note(&g.Q, sh[0], "query rows"); err != nil {
				return g, err
			}
			if err := g.note(&g.D, sh[1], "hidden"); err != nil {
				return g, err
			}
		case "gate", "up":
			if err := g.note(&g.F, sh[0], "intermediate"); err != nil {
				return g, err
			}
			if err := g.note(&g.D, sh[1], "hidden"); err != nil {
				return g, err
			}
		case "embed", "lm_head":
			if err := g.note(&g.Vocab, sh[0], "vocab"); err != nil {
				return g, err
			}
			if err := g.note(&g.D, sh[1], "hidden"); err != nil {
				return g, err
			}
		}
	}
	for _, it := range items {
		if len(it.shape) != 2 {
			continue
		}
		if err := noteSide(&g, assimilRole(it.name), it.shape); err != nil {
			return g, err
		}
	}
	for _, it := range items {
		sh := it.shape
		if len(sh) != 3 || g.D == 0 {
			continue
		}
		e := sh[0]
		switch {
		case strings.Contains(it.name, "gate_up") && sh[1] == g.D && sh[2]%2 == 0:
			if err := g.note(&g.Experts, e, "experts"); err != nil {
				return g, err
			}
			if err := g.note(&g.F, sh[2]/2, "intermediate"); err != nil {
				return g, err
			}
		case strings.Contains(it.name, "gate_up") && sh[2] == g.D && sh[1]%2 == 0:
			if err := g.note(&g.Experts, e, "experts"); err != nil {
				return g, err
			}
			if err := g.note(&g.F, sh[1]/2, "intermediate"); err != nil {
				return g, err
			}
		case strings.Contains(it.name, "down") && sh[2] == g.D:
			if err := g.note(&g.Experts, e, "experts"); err != nil {
				return g, err
			}
			if err := g.note(&g.F, sh[1], "intermediate"); err != nil {
				return g, err
			}
		case strings.Contains(it.name, "down") && sh[1] == g.D:
			if err := g.note(&g.Experts, e, "experts"); err != nil {
				return g, err
			}
			if err := g.note(&g.F, sh[2], "intermediate"); err != nil {
				return g, err
			}
		}
	}
	if g.D == 0 {
		return g, fmt.Errorf("no residual width (no query, key, gate, router, or embedding)")
	}
	return g, nil
}

// noteSide records k, v, o, down, and the router once hidden is known, so a
// transposed matrix does not swap the residual with the head axis.
func noteSide(g *geom, role string, sh []int) error {
	switch role {
	case "k", "v":
		return noteAlong(g, &g.KV, sh, "kv rows")
	case "o":
		return noteAlong(g, &g.Q, []int{sh[1], sh[0]}, "query rows")
	case "down":
		return noteAlong(g, &g.F, []int{sh[1], sh[0]}, "intermediate")
	case "router":
		return noteAlong(g, &g.Experts, sh, "experts")
	}
	return nil
}

func noteAlong(g *geom, axis *int, sh []int, what string) error {
	rows, cols := sh[0], sh[1]
	if g.D > 0 {
		switch {
		case cols == g.D:
			return g.note(axis, rows, what)
		case rows == g.D && cols != g.D:
			return g.note(axis, cols, what)
		default:
			return fmt.Errorf("%s %v does not touch hidden %d", what, sh, g.D)
		}
	}
	if err := g.note(axis, rows, what); err != nil {
		return err
	}
	return g.note(&g.D, cols, "hidden")
}

// assimilRole is roleOf plus the names Llama does not use (gpt-oss routers,
// Grok kernels). The mixer's roleOf is unchanged.
func assimilRole(name string) string {
	if r := roleOf(name); r != "" {
		return r
	}
	n := strings.ToLower(name)
	switch {
	case strings.Contains(n, "query"):
		return "q"
	case strings.Contains(n, "key"):
		return "k"
	case strings.Contains(n, "value"):
		return "v"
	case strings.Contains(n, "router"):
		return "router"
	case strings.Contains(n, "linear_v1"):
		return "gate"
	case strings.Contains(n, "linear_1"):
		return "up"
	case strings.Contains(n, "unembed"), strings.Contains(n, "lm_head"):
		return "lm_head"
	case strings.Contains(n, "embed"):
		return "embed"
	case strings.Contains(n, "attention") && strings.Contains(n, "linear"):
		return "o"
	case strings.Contains(n, "expert") && strings.Contains(n, "/linear"):
		return "down"
	case strings.Contains(n, "gate"):
		return "gate"
	}
	return ""
}

func lengthMap(src, tgt geom) map[int]int {
	m := map[int]int{}
	var ban map[int]bool
	add := func(a, b int) {
		if a <= 0 || b <= 0 {
			return
		}
		if ban[a] {
			return
		}
		if o, ok := m[a]; ok && o != b {
			delete(m, a)
			if ban == nil {
				ban = map[int]bool{}
			}
			ban[a] = true
			return
		}
		m[a] = b
	}
	add(src.D, tgt.D)
	add(src.F, tgt.F)
	add(src.Q, tgt.Q)
	add(src.KV, tgt.KV)
	add(src.Vocab, tgt.Vocab)
	add(src.HD, tgt.HD)
	if tgt.Experts > 0 {
		add(src.Experts, tgt.Experts)
	}
	if src.F > 0 && tgt.F > 0 {
		add(2*src.F, 2*tgt.F)
	}
	return m
}

func orKeep(tgt, src, have int) int {
	if tgt > 0 {
		return tgt
	}
	if have > 0 {
		return have
	}
	return src
}

func targetShape(name string, sh []int, src, tgt geom) ([]int, error) {
	role := assimilRole(name)
	if len(sh) == 2 && src.D > 0 && tgt.D > 0 {
		if s, ok := roleMatrix(role, sh, src, tgt); ok {
			return s, nil
		}
	}
	if len(sh) == 3 && src.D > 0 && tgt.D > 0 {
		if s, ok := expertMatrix(name, sh, src, tgt); ok {
			return s, nil
		}
	}
	lm := lengthMap(src, tgt)
	out := make([]int, len(sh))
	for i, n := range sh {
		if len(sh) == 1 {
			if v := vecTarget(name, n, src, tgt, lm); v > 0 {
				out[i] = v
				continue
			}
		}
		if v, ok := lm[n]; ok {
			out[i] = v
			continue
		}
		out[i] = n
	}
	return out, nil
}

func vecTarget(name string, n int, src, tgt geom, lm map[int]int) int {
	if (strings.Contains(name, "q_norm") || strings.Contains(name, "k_norm")) && n == src.HD && tgt.HD > 0 {
		return tgt.HD
	}
	if v, ok := lm[n]; ok {
		return v
	}
	return n
}

func roleMatrix(role string, sh []int, src, tgt geom) ([]int, bool) {
	d, td := src.D, tgt.D
	switch role {
	case "q":
		return orient(sh, src.Q, d, orKeep(tgt.Q, src.Q, sh[0]), td)
	case "k", "v":
		return orient(sh, src.KV, d, orKeep(tgt.KV, src.KV, sh[0]), td)
	case "o":
		return orient(sh, d, src.Q, td, orKeep(tgt.Q, src.Q, sh[1]))
	case "gate", "up":
		return orient(sh, src.F, d, orKeep(tgt.F, src.F, sh[0]), td)
	case "down":
		return orient(sh, d, src.F, td, orKeep(tgt.F, src.F, sh[1]))
	case "embed", "lm_head":
		return orient(sh, src.Vocab, d, orKeep(tgt.Vocab, src.Vocab, sh[0]), td)
	case "router":
		e := src.Experts
		if e == 0 {
			e = sh[0]
		}
		te := tgt.Experts
		if te == 0 {
			te = e
		}
		return orient(sh, e, d, te, td)
	}
	return nil, false
}

// orient matches [row, col] or, when the two lengths differ, the transpose.
func orient(sh []int, r, c, tr, tc int) ([]int, bool) {
	if r > 0 && c > 0 && sh[0] == r && sh[1] == c {
		return []int{tr, tc}, true
	}
	if r > 0 && c > 0 && r != c && sh[0] == c && sh[1] == r {
		return []int{tc, tr}, true
	}
	// Column is the residual even when this tensor's other axis is not the model's canonical length
	// (a wide MLP layer, a fused pack).
	if c > 0 && sh[1] == c && tc > 0 {
		rows := tr
		if r == 0 || sh[0] != r {
			rows = sh[0]
			if tr > 0 && r > 0 {
				rows = tr
			}
		}
		return []int{rows, tc}, true
	}
	if c > 0 && sh[0] == c && tc > 0 && sh[1] != c {
		cols := tr
		if cols == 0 {
			cols = sh[1]
		}
		return []int{tc, cols}, true
	}
	return nil, false
}

func expertMatrix(name string, sh []int, src, tgt geom) ([]int, bool) {
	e := src.Experts
	if e == 0 {
		e = sh[0]
	}
	te := tgt.Experts
	if te == 0 {
		te = e
	}
	if sh[0] != e && src.Experts > 0 {
		return nil, false
	}
	tf := tgt.F
	if tf == 0 {
		tf = src.F
	}
	switch {
	case strings.Contains(name, "gate_up") && sh[1] == src.D && src.F > 0 && sh[2] == 2*src.F:
		return []int{te, tgt.D, 2 * tf}, true
	case strings.Contains(name, "gate_up") && sh[2] == src.D && src.F > 0 && sh[1] == 2*src.F:
		return []int{te, 2 * tf, tgt.D}, true
	case strings.Contains(name, "down") && sh[1] == src.F && sh[2] == src.D:
		return []int{te, tf, tgt.D}, true
	case strings.Contains(name, "down") && sh[1] == src.D && sh[2] == src.F:
		return []int{te, tgt.D, tf}, true
	}
	return nil, false
}

// ---- plan: tensor, source shard, output shard ----

var borgLayer = regexp.MustCompile(`(?:^|[./_])(?:layers?|blocks?|block|decoder_layer|h)[._](\d+)`)

func layerSpan(name string) (a, b int, ok bool) {
	m := borgLayer.FindStringSubmatchIndex(name)
	if m == nil {
		return 0, 0, false
	}
	return m[2], m[3], true
}

func layerNum(name string) int {
	a, b, ok := layerSpan(name)
	if !ok {
		return -1
	}
	n, err := strconv.Atoi(name[a:b])
	if err != nil {
		return -1
	}
	return n
}

func setLayer(name string, layer int) string {
	a, b, ok := layerSpan(name)
	if !ok {
		return name
	}
	return name[:a] + strconv.Itoa(layer) + name[b:]
}

func suffixKey(name string) string {
	a, b, ok := layerSpan(name)
	if !ok {
		return name
	}
	return name[:a] + "*" + name[b:]
}

func srcLayerFor(t, tgtN, srcN int) int {
	if tgtN <= 1 || srcN <= 1 {
		return 0
	}
	return int(math.Round(float64(t) / float64(tgtN-1) * float64(srcN-1)))
}

func nearestLayer(have []int, want int) int {
	best, bd := have[0], iabs(have[0]-want)
	for _, h := range have[1:] {
		d := iabs(h - want)
		if d < bd || (d == bd && h < best) {
			best, bd = h, d
		}
	}
	return best
}

func iabs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func itoa(n int) string { return strconv.Itoa(n) }

type borgJob struct {
	out, src string
	shard    int
	sh, dst  []int
}

func planBorg(src, tgt geom, cat *catalog) ([]borgJob, error) {
	type pat struct {
		order []int
		at    map[int]srcItem
		plain *srcItem
	}
	pats := map[string]*pat{}
	var keys []string
	for _, it := range cat.items {
		k := suffixKey(it.name)
		p := pats[k]
		if p == nil {
			p = &pat{at: map[int]srcItem{}}
			pats[k] = p
			keys = append(keys, k)
		}
		l := layerNum(it.name)
		if l < 0 {
			cp := it
			p.plain = &cp
			continue
		}
		if _, ok := p.at[l]; !ok {
			p.order = append(p.order, l)
		}
		p.at[l] = it
	}
	sort.Strings(keys)
	var jobs []borgJob
	seen := map[string]bool{}
	add := func(out string, it srcItem, dst []int) error {
		if seen[out] {
			return fmt.Errorf("two tensors write %s", out)
		}
		seen[out] = true
		if len(dst) != len(it.shape) {
			return fmt.Errorf("%s: rank %v to %v", out, it.shape, dst)
		}
		jobs = append(jobs, borgJob{out, it.name, it.shard, it.shape, dst})
		return nil
	}
	for _, k := range keys {
		p := pats[k]
		if p.plain != nil {
			dst, err := targetShape(p.plain.name, p.plain.shape, src, tgt)
			if err != nil {
				return nil, err
			}
			if err := add(p.plain.name, *p.plain, dst); err != nil {
				return nil, err
			}
		}
		if len(p.order) == 0 {
			continue
		}
		sort.Ints(p.order)
		n := tgt.Layers
		if n == 0 {
			n = src.Layers
		}
		if n == 0 {
			n = p.order[len(p.order)-1] + 1
		}
		for t := 0; t < n; t++ {
			want := srcLayerFor(t, n, src.Layers)
			got := nearestLayer(p.order, want)
			it := p.at[got]
			dst, err := targetShape(it.name, it.shape, src, tgt)
			if err != nil {
				return nil, err
			}
			if err := add(setLayer(it.name, t), it, dst); err != nil {
				return nil, err
			}
		}
	}
	sort.Slice(jobs, func(i, j int) bool {
		if jobs[i].shard != jobs[j].shard {
			return jobs[i].shard < jobs[j].shard
		}
		return jobs[i].out < jobs[j].out
	})
	return jobs, nil
}

// ---- write ----

func writeBorg(out string, cat *catalog, jobs []borgJob) error {
	n := len(cat.paths)
	dir, names, err := outputNames(out, n)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	by := map[int][]borgJob{}
	for _, j := range jobs {
		by[j.shard] = append(by[j.shard], j)
	}
	weight := map[string]string{}
	var total int64
	for s := 0; s < n; s++ {
		js := by[s]
		if len(js) == 0 {
			continue
		}
		path := filepath.Join(dir, names[s])
		w, err := NewSTW(path, [][2]string{{"tool", "BorgTensor"}, {"geom", "splineGrid+sampleGrid"}, {"volume", names[s]}})
		if err != nil {
			return err
		}
		for _, j := range js {
			v, sh, err := cat.load(j.src)
			if err != nil {
				w.Abort()
				return fmt.Errorf("%s: %w", j.src, err)
			}
			if !sameInts(sh, j.sh) {
				w.Abort()
				return fmt.Errorf("%s: loaded %v, catalog %v", j.src, sh, j.sh)
			}
			logf("%s shard %d → %s  %v → %v", j.src, s, names[s], j.sh, j.dst)
			shaped, err := shapeND(v, j.sh, j.dst)
			if err != nil {
				w.Abort()
				return fmt.Errorf("%s: %w", j.out, err)
			}
			w.AddBF16(j.out, j.dst, shaped)
			weight[j.out] = names[s]
			total += int64(len(shaped)) * 2
		}
		if err := w.Close(nil); err != nil {
			return err
		}
	}
	if len(names) > 1 {
		idx := map[string]any{"metadata": map[string]any{"total_size": total}, "weight_map": weight}
		b, err := json.MarshalIndent(idx, "", "  ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), append(b, '\n'), 0o644); err != nil {
			return err
		}
	}
	logf("BorgTensor wrote %d tensors, %d bytes, %d shards in %s", len(jobs), total, len(weight), dir)
	return nil
}

func outputNames(out string, n int) (string, []string, error) {
	if n < 1 {
		return "", nil, fmt.Errorf("no shards")
	}
	low := strings.ToLower(out)
	if n == 1 && strings.HasSuffix(low, ".safetensors") {
		return filepath.Dir(out), []string{filepath.Base(out)}, nil
	}
	dir := out
	if strings.HasSuffix(low, ".safetensors") {
		dir = filepath.Dir(out)
	}
	if n == 1 {
		return dir, []string{"model.safetensors"}, nil
	}
	names := make([]string, n)
	for i := range names {
		names[i] = fmt.Sprintf("model-%05d-of-%05d.safetensors", i+1, n)
	}
	return dir, names, nil
}

func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ---- tensor → row shard → column shard → shaped tensor ----

func checkShape(sh, dst []int) error {
	if len(sh) != len(dst) {
		return fmt.Errorf("rank %v to %v", sh, dst)
	}
	if numel(sh) != 0 && len(sh) == 0 {
		return nil
	}
	for _, n := range sh {
		if n <= 0 {
			return fmt.Errorf("shape %v", sh)
		}
	}
	for _, n := range dst {
		if n <= 0 {
			return fmt.Errorf("shape %v", dst)
		}
	}
	return nil
}

// shapeND resamples src onto dst. A trailing matrix whose rows and columns both
// change goes through splineGrid and sampleGrid once. Any other axis uses that
// same path with one column, so an unchanged expert axis is not mixed.
func shapeND(src []float32, sh, dst []int) ([]float32, error) {
	if err := checkShape(sh, dst); err != nil {
		return nil, err
	}
	if numel(sh) != len(src) {
		return nil, fmt.Errorf("len %d, shape %v", len(src), sh)
	}
	if sameInts(sh, dst) {
		return append([]float32(nil), src...), nil
	}
	cur := src
	csh := append([]int(nil), sh...)
	if len(csh) > 2 {
		for ax := 0; ax < len(csh)-2; ax++ {
			if csh[ax] == dst[ax] {
				continue
			}
			cur = resampleAxis(cur, csh, ax, dst[ax])
			csh[ax] = dst[ax]
		}
	}
	if len(csh) >= 2 {
		r, c := len(csh)-2, len(csh)-1
		switch {
		case csh[r] != dst[r] && csh[c] != dst[c]:
			return shapeBatched(cur, csh, dst), nil
		case csh[r] != dst[r]:
			return resampleAxis(cur, csh, r, dst[r]), nil
		case csh[c] != dst[c]:
			return resampleAxis(cur, csh, c, dst[c]), nil
		}
		return cur, nil
	}
	if csh[0] != dst[0] {
		return resampleAxis(cur, csh, 0, dst[0]), nil
	}
	return cur, nil
}

func shapeBatched(src []float32, sh, dst []int) []float32 {
	r, c := len(sh)-2, len(sh)-1
	R, C, Rt, Ct := sh[r], sh[c], dst[r], dst[c]
	batch := numel(sh[:r])
	out := make([]float32, batch*Rt*Ct)
	ps, pd := R*C, Rt*Ct
	for b := 0; b < batch; b++ {
		copy(out[b*pd:(b+1)*pd], shapeJoint(src[b*ps:(b+1)*ps], R, C, Rt, Ct))
	}
	return out
}

// shapeJoint is the Llama reshape: linCoords, splineGrid, sampleGrid, one matrix.
func shapeJoint(src []float32, R, C, Rt, Ct int) []float32 {
	if Rt == R && Ct == C {
		return append([]float32(nil), src...)
	}
	xr := linCoords(R)
	xc := linCoords(C)
	G := [2]int{gridSize(R, 0), gridSize(C, 0)}
	grid := splineGrid(src, R, C, xr, xc, G, 0)
	out := make([]float32, Rt*Ct)
	yc := linCoords(Ct)
	yr := linCoords(Rt)
	const chunk = 1024
	for i0 := 0; i0 < Rt; i0 += chunk {
		i1 := i0 + chunk
		if i1 > Rt {
			i1 = Rt
		}
		samp := sampleGrid(grid, G, yr[i0:i1], yc)
		for i, v := range samp {
			out[i0*Ct+i] = float32(v)
		}
	}
	return out
}

// resampleAxis is the one-column form of shapeJoint. The column shard has
// length 1, which sampleGrid reads back unchanged, and the row shard is the
// smoothed 1-form.
func resampleAxis(src []float32, sh []int, ax, m int) []float32 {
	n := sh[ax]
	inner := numel(sh[ax+1:])
	outer := numel(sh[:ax])
	dst := make([]float32, outer*m*inner)
	xs := linCoords(n)
	G := gridSize(n, 0)
	geom := newAxisGeom(xs, G)
	gr := make([]float64, G)
	for i := range gr {
		gr[i] = (float64(i) + 0.5) / float64(G)
	}
	yr := linCoords(m)
	col := make([]float64, n)
	for o := 0; o < outer; o++ {
		for in := 0; in < inner; in++ {
			base := o*n*inner + in
			for i := 0; i < n; i++ {
				col[i] = float64(src[base+i*inner])
			}
			proj := geom.project(col)
			ob := o*m*inner + in
			for j := 0; j < m; j++ {
				dst[ob+j*inner] = float32(interp(yr[j], gr, proj))
			}
		}
	}
	return dst
}

func linCoords(n int) []float64 {
	x := make([]float64, n)
	if n <= 1 {
		if n == 1 {
			x[0] = 0.5
		}
		return x
	}
	for i := range x {
		x[i] = (float64(i) + 0.5) / float64(n)
	}
	return x
}
