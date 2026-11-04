package main

import (
	"fmt"
	"math"
)

// fp4 is the E2M1 table, nibble order. Index 8 is −0.
var fp4 = [16]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}

// DequantMXFP4 unpacks blocks [..., rows, groups, bytes] and scales [..., rows, groups] into
// [..., cols, rows] with cols = groups * bytes * 2. The last two axes are transposed, which is
// the float checkpoint's layout.
func DequantMXFP4(blocks, scales []byte, bShape, sShape []int) ([]float32, []int, error) {
	if len(bShape) < 3 || len(sShape) != len(bShape)-1 {
		return nil, nil, fmt.Errorf("mxfp4 shapes blocks %v scales %v", bShape, sShape)
	}
	for i := range sShape {
		if bShape[i] != sShape[i] {
			return nil, nil, fmt.Errorf("mxfp4 blocks %v do not match scales %v", bShape, sShape)
		}
	}
	B := bShape[len(bShape)-1]
	groups := bShape[len(bShape)-2]
	rows := 1
	for _, v := range bShape[:len(bShape)-2] {
		rows *= v
	}
	if B < 1 || groups < 1 || rows < 1 || len(blocks) != rows*groups*B || len(scales) != rows*groups {
		return nil, nil, fmt.Errorf("mxfp4 bytes blocks %d scales %d, shapes %v %v", len(blocks), len(scales), bShape, sShape)
	}
	cols := groups * B * 2
	raw := make([]float32, rows*cols) // [rows, cols] before the last-two transpose
	for r := 0; r < rows; r++ {
		for g := 0; g < groups; g++ {
			exp := int(scales[r*groups+g]) - 127
			blk := blocks[(r*groups+g)*B : (r*groups+g+1)*B]
			dst := raw[r*cols+g*B*2:]
			for j, b := range blk {
				dst[2*j] = float32(math.Ldexp(float64(fp4[b&0x0f]), exp))
				dst[2*j+1] = float32(math.Ldexp(float64(fp4[b>>4]), exp))
			}
		}
	}
	// bShape is [prefix..., rowAxis, groups, bytes]. `rows` above folds prefix into rowAxis.
	rowAxis := bShape[len(bShape)-3]
	if rowAxis < 1 || rows%rowAxis != 0 {
		return nil, nil, fmt.Errorf("mxfp4 row axis %d does not divide %d", rowAxis, rows)
	}
	pre := rows / rowAxis
	out := make([]float32, len(raw))
	for p := 0; p < pre; p++ {
		for r := 0; r < rowAxis; r++ {
			for c := 0; c < cols; c++ {
				out[(p*cols+c)*rowAxis+r] = raw[(p*rowAxis+r)*cols+c]
			}
		}
	}
	osh := make([]int, 0, len(bShape)-1)
	osh = append(osh, bShape[:len(bShape)-3]...)
	osh = append(osh, cols, rowAxis)
	return out, osh, nil
}
