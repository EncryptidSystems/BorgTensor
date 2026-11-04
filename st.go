package main

// safetensors reader (ordered header, offset reads) and two-pass writer (temp file + header at close).

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
)

type TInfo struct {
	Dtype string  `json:"dtype"`
	Shape []int   `json:"shape"`
	Off   []int64 `json:"data_offsets"`
}

type ST struct {
	Path  string
	f     *os.File
	base  int64
	Keys  []string
	H     map[string]TInfo
	Meta  [][2]string
	owner map[string]*ST // sharded checkpoint: the shard that holds each tensor (nil for one file)
}

// Paper: no equation (helper: I/O, progress or utility)
func openST(path string) (*ST, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var hb [8]byte
	if _, err := f.ReadAt(hb[:], 0); err != nil {
		return nil, err
	}
	n := int64(binary.LittleEndian.Uint64(hb[:]))
	hj := make([]byte, n)
	if _, err := f.ReadAt(hj, 8); err != nil {
		return nil, err
	}
	s := &ST{Path: path, f: f, base: 8 + n, H: map[string]TInfo{}}
	dec := json.NewDecoder(bytes.NewReader(hj))
	if _, err := dec.Token(); err != nil {
		return nil, err
	}
	for dec.More() {
		t, err := dec.Token()
		if err != nil {
			return nil, err
		}
		k := t.(string)
		if k == "__metadata__" {
			var m map[string]any
			var raw json.RawMessage
			dec.Decode(&raw)
			json.Unmarshal(raw, &m)
			// keep header order
			d2 := json.NewDecoder(bytes.NewReader(raw))
			d2.Token()
			for d2.More() {
				kt, _ := d2.Token()
				var v any
				d2.Decode(&v)
				s.Meta = append(s.Meta, [2]string{kt.(string), fmt.Sprint(v)})
			}
			continue
		}
		var ti TInfo
		if err := dec.Decode(&ti); err != nil {
			return nil, err
		}
		s.Keys = append(s.Keys, k)
		s.H[k] = ti
	}
	return s, nil
}

// Paper: no equation (helper: I/O, progress or utility)
func (s *ST) Has(k string) bool { _, ok := s.H[k]; return ok }

// Close releases the safetensors files this ST holds.
func (s *ST) Close() error {
	if s == nil {
		return nil
	}
	var err error
	if s.f != nil {
		err = s.f.Close()
		s.f = nil
	}
	seen := map[*ST]bool{}
	for _, o := range s.owner {
		if o == nil || o == s || seen[o] {
			continue
		}
		seen[o] = true
		if o.f != nil {
			if e := o.f.Close(); err == nil {
				err = e
			}
			o.f = nil
		}
	}
	return err
}

// Paper: no equation (helper: I/O, progress or utility)
func (s *ST) Raw(k string) ([]byte, error) {
	if o := s.owner[k]; o != nil {
		return o.Raw(k)
	}
	t, ok := s.H[k]
	if !ok {
		return nil, fmt.Errorf("%s: no tensor %s", s.Path, k)
	}
	b := make([]byte, t.Off[1]-t.Off[0])
	_, err := s.f.ReadAt(b, s.base+t.Off[0])
	return b, err
}

// Paper: no equation (helper: I/O, progress or utility)
func numel(sh []int) int {
	n := 1
	for _, v := range sh {
		n *= v
	}
	return n
}

// Paper: no equation (helper: I/O, progress or utility)
func f16to32(h uint16) float32 {
	sgn := uint32(h>>15) << 31
	e := int((h >> 10) & 31)
	m := uint32(h & 1023)
	switch {
	case e == 0:
		if m == 0 {
			return math.Float32frombits(sgn)
		}
		v := float32(m) / 1024 / 16384
		if sgn != 0 {
			v = -v
		}
		return v
	case e == 31:
		return math.Float32frombits(sgn | 0x7f800000 | m<<13)
	}
	return math.Float32frombits(sgn | uint32(e-15+127)<<23 | m<<13)
}

func f32to16(f float32) uint16 { // round-to-nearest-even, as numpy astype(float16)
	b := math.Float32bits(f)
	sgn := uint16(b>>16) & 0x8000
	e := int((b>>23)&255) - 127 + 15
	m := b & 0x7fffff
	if (b>>23)&255 == 255 {
		if m != 0 {
			return sgn | 0x7e00
		}
		return sgn | 0x7c00
	}
	if e >= 31 {
		return sgn | 0x7c00
	}
	if e <= 0 {
		if e < -10 {
			return sgn
		}
		m |= 0x800000
		sh := uint32(14 - e)
		v := m >> sh
		rem := m & (1<<sh - 1)
		half := uint32(1) << (sh - 1)
		if rem > half || (rem == half && v&1 == 1) {
			v++
		}
		return sgn | uint16(v)
	}
	v := uint32(e)<<10 | m>>13
	rem := m & 0x1fff
	if rem > 0x1000 || (rem == 0x1000 && v&1 == 1) {
		v++
	}
	return sgn | uint16(v)
}

func bf16(f float32) uint16 {
	u := math.Float32bits(f)
	return uint16((u + 0x7FFF + ((u >> 16) & 1)) >> 16)
}

// F32 reads a float tensor as float32 (BF16/F16/F32/F64).
// Paper: no equation (helper: I/O, progress or utility)
func (s *ST) F32(k string) ([]float32, error) {
	b, err := s.Raw(k)
	if err != nil {
		return nil, err
	}
	t := s.H[k]
	n := numel(t.Shape)
	o := make([]float32, n)
	switch t.Dtype {
	case "BF16":
		for i := range o {
			o[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*i:])) << 16)
		}
	case "F16":
		for i := range o {
			o[i] = f16to32(binary.LittleEndian.Uint16(b[2*i:]))
		}
	case "F32":
		for i := range o {
			o[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
	case "F64":
		for i := range o {
			o[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(b[8*i:])))
		}
	case "U8":
		for i := range o {
			o[i] = float32(b[i])
		}
	case "I8":
		for i := range o {
			o[i] = float32(int8(b[i]))
		}
	default:
		return nil, fmt.Errorf("%s: dtype %s not float", k, t.Dtype)
	}
	return o, nil
}

// Paper: no equation (helper: I/O, progress or utility)
func scaleKey(k string) string { return strings.TrimSuffix(k, "weight") + "weight_scale" }

// Paper: no equation (helper: I/O, progress or utility)
func (s *ST) isPacked(k string) bool {
	t := s.H[k]
	return t.Dtype == "U8" && strings.HasSuffix(k, ".weight") && s.Has(scaleKey(k))
}

// Paper: no equation (helper: I/O, progress or utility)
func (s *ST) LogicalShape(k string) []int {
	t := s.H[k]
	if s.isPacked(k) {
		return append([]int{t.Shape[0] * 4}, t.Shape[1:]...)
	}
	return t.Shape
}

// LoadDense: float32 weight; packed BitNet weights (U8 + weight_scale) are dequantized (t-1)*scale (Champ/edgellm convention).
// Paper: Sec.4 base is ternary: dequantize (t-1)*weight_scale (BitNet b1.58 2B4T autobitlinear; edgellm bitlinear.wgsl)
func (s *ST) LoadDense(k string) ([]float32, []int, error) {
	if s.isPacked(k) {
		sc, err := s.F32(scaleKey(k))
		if err != nil {
			return nil, nil, err
		}
		b, err := s.Raw(k)
		if err != nil {
			return nil, nil, err
		}
		sh := s.H[k].Shape
		R4 := sh[0]
		C := numel(sh[1:])
		o := make([]float32, 4*len(b))
		for sI := 0; sI < 4; sI++ {
			for i, v := range b {
				o[sI*R4*C+i] = float32(int((v>>(2*sI))&3)-1) * sc[0]
			}
		}
		for _, lb := range s.limbsOf(k) { // dynamic MRH limbs (limbs.go): the weight is their sum
			c := (lb.r - 1) / 2
			for i := range o {
				o[i] += float32(lb.s * float64(int(lb.d[i])-c))
			}
		}
		return o, s.LogicalShape(k), nil
	}
	v, err := s.F32(k)
	return v, s.H[k].Shape, err
}

// ---------------- writer ----------------
type STW struct {
	path string
	tmp  *os.File
	bw   *bufio.Writer
	keys []string
	h    map[string]TInfo
	off  int64
	Meta [][2]string
}

// Paper: no equation (helper: I/O, progress or utility)
func NewSTW(path string, meta [][2]string) (*STW, error) {
	t, err := os.CreateTemp(filepath.Dir(path), ".borgtensor-*.tmp")
	if err != nil {
		return nil, err
	}
	tempMu.Lock()
	tempFiles[t.Name()] = t
	tempMu.Unlock()
	return &STW{path: path, tmp: t, bw: bufio.NewWriterSize(t, 1<<20), h: map[string]TInfo{}, Meta: meta}, nil
}

// Temp files of writers that have not closed; removeTemps deletes them when a run stops early
// (die, Ctrl+C), so an interrupted run leaves no .borgtensor-*.tmp behind.
var (
	tempMu    sync.Mutex
	tempFiles = map[string]*os.File{}
)

func removeTemps() {
	tempMu.Lock()
	defer tempMu.Unlock()
	for p, f := range tempFiles {
		_ = f.Close()
		_ = os.Remove(p)
		delete(tempFiles, p)
	}
}

// Patch overwrites the bytes of a tensor already added (same length), before Close.
func (w *STW) Patch(k string, b []byte) error {
	t, ok := w.h[k]
	if !ok || int64(len(b)) != t.Off[1]-t.Off[0] {
		return fmt.Errorf("patch %s: no tensor of %d bytes", k, len(b))
	}
	if err := w.bw.Flush(); err != nil {
		return err
	}
	_, err := w.tmp.WriteAt(b, t.Off[0])
	return err
}

// Paper: no equation (helper: I/O, progress or utility)
func (w *STW) Add(k, dtype string, shape []int, b []byte) {
	tw := w.Append(k, dtype, shape)
	_, _ = tw.Write(b)
}

// Append starts tensor k. Bytes written to the result are the tensor. The next
// Add or Append begins the following tensor.
func (w *STW) Append(k, dtype string, shape []int) io.Writer {
	if shape == nil {
		shape = []int{}
	}
	w.keys = append(w.keys, k)
	w.h[k] = TInfo{dtype, shape, []int64{w.off, w.off}}
	return &tensorWriter{w: w, k: k}
}

type tensorWriter struct {
	w *STW
	k string
}

func (t *tensorWriter) Write(b []byte) (int, error) {
	if _, err := t.w.bw.Write(b); err != nil {
		return 0, err
	}
	t.w.off += int64(len(b))
	inf := t.w.h[t.k]
	inf.Off[1] = t.w.off
	t.w.h[t.k] = inf
	return len(b), nil
}

// Abort drops a writer that will not be closed.
func (w *STW) Abort() {
	if w.tmp == nil {
		return
	}
	_ = w.tmp.Close()
	_ = os.Remove(w.tmp.Name())
	tempMu.Lock()
	delete(tempFiles, w.tmp.Name())
	tempMu.Unlock()
	w.tmp = nil
}

// Paper: no equation (helper: I/O, progress or utility)
func (w *STW) AddF32(k string, shape []int, a []float32) {
	b := make([]byte, 4*len(a))
	for i, v := range a {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	w.Add(k, "F32", shape, b)
}

// Paper: no equation (helper: I/O, progress or utility)
func (w *STW) AddI32(k string, shape []int, a []int32) {
	b := make([]byte, 4*len(a))
	for i, v := range a {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(v))
	}
	w.Add(k, "I32", shape, b)
}

// Paper: no equation (helper: I/O, progress or utility)
func (w *STW) AddBF16(k string, shape []int, a []float32) {
	b := make([]byte, 2*len(a))
	for i, v := range a {
		binary.LittleEndian.PutUint16(b[2*i:], bf16(v))
	}
	w.Add(k, "BF16", shape, b)
}

// Paper: no equation (helper: I/O, progress or utility)
func (w *STW) AddF16(k string, shape []int, a []float32) {
	b := make([]byte, 2*len(a))
	for i, v := range a {
		binary.LittleEndian.PutUint16(b[2*i:], f32to16(v))
	}
	w.Add(k, "F16", shape, b)
}

// Paper: no equation (helper: I/O, progress or utility)

// Paper: no equation (helper: I/O, progress or utility)
func (w *STW) Close(prog func(done, tot int64)) error {
	w.bw.Flush()
	var hb bytes.Buffer
	hb.WriteString("{")
	for _, k := range w.keys {
		kj, _ := json.Marshal(k)
		vj, _ := json.Marshal(w.h[k])
		hb.Write(kj)
		hb.WriteString(":")
		hb.Write(vj)
		hb.WriteString(",")
	}
	hb.WriteString(`"__metadata__":{`)
	for i, kv := range w.Meta {
		if i > 0 {
			hb.WriteString(",")
		}
		a, _ := json.Marshal(kv[0])
		b, _ := json.Marshal(kv[1])
		hb.Write(a)
		hb.WriteString(":")
		hb.Write(b)
	}
	hb.WriteString("}}")
	for hb.Len()%8 != 0 {
		hb.WriteString(" ")
	}
	f, err := os.Create(w.path)
	if err != nil {
		return err
	}
	var nb [8]byte
	binary.LittleEndian.PutUint64(nb[:], uint64(hb.Len()))
	f.Write(nb[:])
	f.Write(hb.Bytes())
	w.tmp.Seek(0, 0)
	buf := make([]byte, 1<<22)
	var done int64
	for {
		n, err := w.tmp.Read(buf)
		if n > 0 {
			f.Write(buf[:n])
			done += int64(n)
			if prog != nil {
				prog(done, w.off)
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}
	w.tmp.Close()
	os.Remove(w.tmp.Name())
	tempMu.Lock()
	delete(tempFiles, w.tmp.Name())
	tempMu.Unlock()
	return f.Close()
}

type limbData struct {
	r int
	d []uint8
	s float64
}

func limbName(k string, l int) string { return k + "_limb" + strconv.Itoa(l) }

func unpackDigits(b []byte, R, C int) []uint8 {
	R4 := R / 4
	d := make([]uint8, R*C)
	for i := 0; i < R4*C; i++ {
		for s := 0; s < 4; s++ {
			d[s*R4*C+i] = (b[i] >> (2 * s)) & 3
		}
	}
	return d
}

// limbsOf reads tensor k's extra limbs from s (radix, digits, scale).
func (s *ST) limbsOf(k string) []limbData {
	var out []limbData
	if !s.Has(k) {
		return nil
	}
	sh := s.LogicalShape(k)
	if len(sh) != 2 {
		return nil
	}
	for l := 1; ; l++ {
		n := limbName(k, l)
		if !s.Has(n) || !s.Has(n+"_scale") || !s.Has(n+"_radix") {
			return out
		}
		b, err1 := s.Raw(n)
		sc, err2 := s.F32(n + "_scale")
		rb, err3 := s.Raw(n + "_radix")
		if err1 != nil || err2 != nil || err3 != nil || len(sc) != 1 || len(rb) != 4 {
			return out
		}
		out = append(out, limbData{int(binary.LittleEndian.Uint32(rb)), unpackDigits(b, sh[0], sh[1]), float64(sc[0])})
	}
}

// isLimbTensor reports a limb tensor (or its scale / radix) name.
func isLimbTensor(n string) bool {
	for _, suf := range []string{"_scale", "_radix", ""} {
		m := n[:len(n)-len(suf)]
		if len(n) < len(suf) || n[len(n)-len(suf):] != suf {
			continue
		}
		i := len(m) - 1
		for i >= 0 && m[i] >= '0' && m[i] <= '9' {
			i--
		}
		if i < len(m)-1 && i >= 11 && m[i-11:i+1] == ".weight_limb" {
			return true
		}
	}
	return false
}
