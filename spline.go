package main

import (
	"math"
	"runtime"
	"sort"
	"sync"
)

// bucketBits is paper_v2 Eq.(6): two MRH levels of 16 buckets, grid cap 256.
const bucketBits = 4

func radices(n int) []int {
	var out []int
	m := n
	for p := 2; p*p <= m; p++ {
		for m%p == 0 {
			out = append(out, p)
			m /= p
		}
	}
	if m > 1 {
		out = append(out, m)
	}
	if len(out) == 0 {
		return []int{1}
	}
	sort.Sort(sort.Reverse(sort.IntSlice(out)))
	return out
}

func gridSize(n, _ int) int {
	g := 1
	for _, r := range radices(n) {
		if g*r > 1<<(2*bucketBits) {
			break
		}
		g *= r
	}
	return max(g, 1)
}

func parFor(n int, f func(lo, hi int)) {
	w := runtime.NumCPU()
	if n < 64 || w == 1 {
		f(0, n)
		return
	}
	var wg sync.WaitGroup
	ch := (n + w - 1) / w
	for lo := 0; lo < n; lo += ch {
		hi := min(n, lo+ch)
		wg.Add(1)
		go func(lo, hi int) { defer wg.Done(); f(lo, hi) }(lo, hi)
	}
	wg.Wait()
}

func interp(x float64, xp, fp []float64) float64 {
	n := len(xp)
	if x <= xp[0] {
		return fp[0]
	}
	if x >= xp[n-1] {
		return fp[n-1]
	}
	j := sort.SearchFloat64s(xp, x) // first xp[j] >= x
	if xp[j] == x {
		return fp[j]
	}
	i := j - 1
	t := (x - xp[i]) / (xp[j] - xp[i])
	return fp[i] + t*(fp[j]-fp[i])
}

func argsortStable(x []float64) []int {
	o := make([]int, len(x))
	for i := range o {
		o[i] = i
	}
	sort.SliceStable(o, func(a, b int) bool { return x[o[a]] < x[o[b]] })
	return o
}

// splineGrid resamples the channel map onto the index grid.
// Paper: Contrib.4. Each axis is the smoothed 1-form of fec.md §6.
func splineGrid(W []float32, R, C int, xr, xc []float64, G [2]int, jitter float64) []float64 {
	o := argsortStable(xr)
	xs := make([]float64, R)
	for i, k := range o {
		xs[i] = xr[k] + float64(i)*jitter
	}
	A := make([]float64, G[0]*C) // G0 x C
	rowGeom := newAxisGeom(xs, G[0])
	parFor(C, func(lo, hi int) {
		col := make([]float64, R)
		for j := lo; j < hi; j++ {
			for i, k := range o {
				col[i] = float64(W[k*C+j])
			}
			proj := rowGeom.project(col)
			for g := 0; g < G[0]; g++ {
				A[g*C+j] = proj[g]
			}
		}
	})
	oc := argsortStable(xc)
	xcs := make([]float64, C)
	for i, k := range oc {
		xcs[i] = xc[k] + float64(i)*jitter
	}
	out := make([]float64, G[0]*G[1])
	colGeom := newAxisGeom(xcs, G[1])
	parFor(G[0], func(lo, hi int) {
		row := make([]float64, C)
		for g := lo; g < hi; g++ {
			for i, k := range oc {
				row[i] = A[g*C+k]
			}
			proj := colGeom.project(row)
			for h := 0; h < G[1]; h++ {
				out[g*G[1]+h] = proj[h]
			}
		}
	})
	return out
}

// Smoothed 1-form on one axis (fec.md §6). δ = 1, so the mollifier half-width equals one coarse cell.
// The 0-form is not implemented. Outside [0,1] the slope keeps the value it has at that endpoint,
// which is what sends a straight line back to that line. The reported sample is the midpoint of the
// coarse edge. The coboundary does not wrap. xs must be sorted.
type axisGeom struct {
	xs     []float64
	G      int
	h      float64
	sL, sR int // edge whose slope extends past 0 and past 1; -1 means that slope is 0
	left   []float64
	right  []float64
	seg    []axisSeg
	lu     []float64
	piv    []int
}

// axisSeg holds one fine edge clipped to [0,1]. w[k] is the cell integral of the mollified
// indicator on coarse cell c0+k. Cells farther than ε from the edge are omitted.
type axisSeg struct {
	edge int
	c0   int
	w    []float64
}

func newAxisGeom(xs []float64, G int) axisGeom {
	g := axisGeom{xs: append([]float64(nil), xs...), G: G, h: 1 / float64(G), sL: -1, sR: -1}
	if G <= 0 {
		return g
	}
	g.left = make([]float64, G)
	g.right = make([]float64, G)
	n := len(xs)
	if n < 2 {
		return g
	}
	eps := g.h
	for i := 0; i < n-1; i++ {
		a0, b0 := xs[i], xs[i+1]
		if b0 <= a0 || b0 <= 0 || a0 >= 1 {
			continue
		}
		a, b := a0, b0
		if a < 0 {
			a = 0
		}
		if b > 1 {
			b = 1
		}
		c0 := int(math.Floor((a - eps) / g.h))
		if c0 < 0 {
			c0 = 0
		}
		c1 := int(math.Ceil((b + eps) / g.h))
		if c1 > G {
			c1 = G
		}
		if c1 <= c0 {
			continue
		}
		w := make([]float64, c1-c0)
		for c := c0; c < c1; c++ {
			w[c-c0] = mollWeight(float64(c)*g.h, float64(c+1)*g.h, a, b, eps)
		}
		g.seg = append(g.seg, axisSeg{edge: i, c0: c0, w: w})
	}
	if xs[0] <= 0 {
		for i := 0; i < n-1; i++ {
			if xs[i] <= 0 && xs[i+1] > 0 {
				g.sL = i
				break
			}
		}
	}
	if xs[n-1] >= 1 {
		for i := n - 2; i >= 0; i-- {
			if xs[i] < 1 && xs[i+1] >= 1 {
				g.sR = i
				break
			}
		}
	}
	for c := 0; c < G; c++ {
		L, R := float64(c)*g.h, float64(c+1)*g.h
		g.left[c] = mollRayLeft(L, R, eps)
		g.right[c] = mollRayRight(L, R, eps)
	}
	g.factorMass()
	return g
}

// factorMass is the LU of Q restricted to the coarse 1-forms. Basis cell c is the indicator of
// that cell; cell 0 also carries the left ray and the last cell the right ray, which is the
// constant extension of the endpoint slope.
func (g *axisGeom) factorMass() {
	G := g.G
	m := make([]float64, G*G)
	for c := 0; c < G; c++ {
		a := float64(c) * g.h
		b := a + g.h
		for r := 0; r < G; r++ {
			L := float64(r) * g.h
			R := L + g.h
			w := mollWeight(L, R, a, b, g.h)
			if c == 0 {
				w += g.left[r]
			}
			if c == G-1 {
				w += g.right[r]
			}
			m[r*G+c] = w
		}
	}
	piv := make([]int, G)
	for k := 0; k < G; k++ {
		p := k
		best := math.Abs(m[k*G+k])
		for i := k + 1; i < G; i++ {
			if v := math.Abs(m[i*G+k]); v > best {
				best, p = v, i
			}
		}
		piv[k] = p
		if p != k {
			for j := 0; j < G; j++ {
				m[k*G+j], m[p*G+j] = m[p*G+j], m[k*G+j]
			}
		}
		pivot := m[k*G+k]
		if pivot == 0 {
			continue
		}
		for i := k + 1; i < G; i++ {
			f := m[i*G+k] / pivot
			m[i*G+k] = f
			for j := k + 1; j < G; j++ {
				m[i*G+j] -= f * m[k*G+j]
			}
		}
	}
	g.lu = m
	g.piv = piv
}

func (g axisGeom) solve(raw []float64) []float64 {
	G := g.G
	y := append([]float64(nil), raw...)
	for k := 0; k < G; k++ {
		if p := g.piv[k]; p != k {
			y[k], y[p] = y[p], y[k]
		}
		for i := k + 1; i < G; i++ {
			y[i] -= g.lu[i*G+k] * y[k]
		}
	}
	for k := G - 1; k >= 0; k-- {
		s := y[k]
		for j := k + 1; j < G; j++ {
			s -= g.lu[k*G+j] * y[j]
		}
		if d := g.lu[k*G+k]; d == 0 {
			y[k] = 0
		} else {
			y[k] = s / d
		}
	}
	return y
}

func (g axisGeom) slope(us []float64, edge int) float64 {
	dx := g.xs[edge+1] - g.xs[edge]
	if dx == 0 || edge < 0 || edge+1 >= len(us) {
		return 0
	}
	return (us[edge+1] - us[edge]) / dx
}

func (g axisGeom) atZero(us []float64) float64 {
	xs := g.xs
	if len(xs) == 0 || xs[0] >= 0 {
		return us[0]
	}
	for i := 0; i < len(xs)-1 && i+1 < len(us); i++ {
		if xs[i] <= 0 && xs[i+1] > 0 {
			dx := xs[i+1] - xs[i]
			if dx == 0 {
				return us[i]
			}
			return us[i] + (0-xs[i])/dx*(us[i+1]-us[i])
		}
	}
	return us[len(us)-1]
}

func (g axisGeom) project(us []float64) []float64 {
	out := make([]float64, g.G)
	if g.G == 0 || len(us) == 0 {
		return out
	}
	if len(us) < 2 || len(g.xs) < 2 || g.lu == nil {
		for i := range out {
			out[i] = us[0]
		}
		return out
	}
	raw := make([]float64, g.G)
	if g.sL >= 0 {
		s := g.slope(us, g.sL)
		for c := 0; c < g.G; c++ {
			raw[c] += s * g.left[c]
		}
	}
	if g.sR >= 0 {
		s := g.slope(us, g.sR)
		for c := 0; c < g.G; c++ {
			raw[c] += s * g.right[c]
		}
	}
	for _, seg := range g.seg {
		s := g.slope(us, seg.edge)
		for k, w := range seg.w {
			raw[seg.c0+k] += s * w
		}
	}
	sigma := g.solve(raw)
	v := g.atZero(us)
	for c := 0; c < g.G; c++ {
		inc := sigma[c] * g.h
		out[c] = v + 0.5*inc
		v += inc
	}
	return out
}

// bump is φ(t) = exp(-1/(1-t²)) on (-1,1). Φ and Ψ are its first two antiderivatives from -1.
const bumpN = 8192

var (
	bumpOnce   sync.Once
	bumpPhiTab []float64
	bumpPsiTab []float64
	bumpZ      float64
)

func bumpInit() {
	bumpOnce.Do(func() {
		phi := make([]float64, bumpN+1)
		dt := 2.0 / float64(bumpN)
		for i := 0; i <= bumpN; i++ {
			t := -1 + dt*float64(i)
			if t > -1 && t < 1 {
				phi[i] = math.Exp(-1 / (1 - t*t))
			}
		}
		bumpPhiTab = make([]float64, bumpN+1)
		bumpPsiTab = make([]float64, bumpN+1)
		for i := 1; i <= bumpN; i++ {
			bumpPhiTab[i] = bumpPhiTab[i-1] + 0.5*(phi[i]+phi[i-1])*dt
			bumpPsiTab[i] = bumpPsiTab[i-1] + 0.5*(bumpPhiTab[i]+bumpPhiTab[i-1])*dt
		}
		bumpZ = bumpPhiTab[bumpN]
	})
}

func bumpPhi(t float64) float64 {
	bumpInit()
	if t <= -1 {
		return 0
	}
	if t >= 1 {
		return bumpZ
	}
	p := (t + 1) * 0.5 * float64(bumpN)
	i := int(p)
	if i >= bumpN {
		return bumpZ
	}
	f := p - float64(i)
	return bumpPhiTab[i]*(1-f) + bumpPhiTab[i+1]*f
}

func bumpPsi(t float64) float64 {
	bumpInit()
	if t <= -1 {
		return 0
	}
	if t >= 1 {
		return bumpPsiTab[bumpN] + bumpZ*(t-1)
	}
	p := (t + 1) * 0.5 * float64(bumpN)
	i := int(p)
	if i >= bumpN {
		i = bumpN - 1
	}
	f := p - float64(i)
	return bumpPsiTab[i]*(1-f) + bumpPsiTab[i+1]*f
}

// mollWeight is ∫_L^R K_ε(1_[a,b]) dx. K_ε is the unit-mass mollifier at scale ε,
// so the integral is (ε/Z) * ((Ψ((R-a)/ε)-Ψ((L-a)/ε)) - (Ψ((R-b)/ε)-Ψ((L-b)/ε))).
func mollWeight(L, R, a, b, eps float64) float64 {
	if !(R > L) || !(b > a) || eps <= 0 {
		return 0
	}
	bumpInit()
	return eps / bumpZ * ((bumpPsi((R-a)/eps) - bumpPsi((L-a)/eps)) - (bumpPsi((R-b)/eps) - bumpPsi((L-b)/eps)))
}

func mollRayLeft(L, R, eps float64) float64 {
	bumpInit()
	return (R - L) - eps/bumpZ*(bumpPsi(R/eps)-bumpPsi(L/eps))
}

func mollRayRight(L, R, eps float64) float64 {
	bumpInit()
	return eps / bumpZ * (bumpPsi((R-1)/eps) - bumpPsi((L-1)/eps))
}

// Paper: Contrib.4 / Sec.4 resize: sample the index spline at the target channels
func sampleGrid(S []float64, G [2]int, xr, xc []float64) []float64 {
	gr := make([]float64, G[0])
	for i := range gr {
		gr[i] = (float64(i) + .5) / float64(G[0])
	}
	gc := make([]float64, G[1])
	for i := range gc {
		gc[i] = (float64(i) + .5) / float64(G[1])
	}
	R, C := len(xr), len(xc)
	A := make([]float64, R*G[1])
	parFor(G[1], func(lo, hi int) {
		col := make([]float64, G[0])
		for j := lo; j < hi; j++ {
			for g := 0; g < G[0]; g++ {
				col[g] = S[g*G[1]+j]
			}
			for i := 0; i < R; i++ {
				A[i*G[1]+j] = interp(xr[i], gr, col)
			}
		}
	})
	out := make([]float64, R*C)
	parFor(R, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			row := A[i*G[1] : (i+1)*G[1]]
			for c := 0; c < C; c++ {
				out[i*C+c] = interp(xc[c], gc, row)
			}
		}
	})
	return out
}
