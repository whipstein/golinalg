package matgen

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

const (
	Full      = mat.Full
	Lower     = mat.Lower
	Upper     = mat.Upper
	Trans     = mat.Trans
	NoTrans   = mat.NoTrans
	ConjTrans = mat.ConjTrans
	Left      = mat.Left
	Right     = mat.Right
	NonUnit   = mat.NonUnit
	Unit      = mat.Unit
)

const (
	Epsilon     = golapack.Epsilon
	Overflow    = golapack.Overflow
	Underflow   = golapack.Underflow
	SafeMinimum = golapack.SafeMinimum
	Precision   = golapack.Precision
	Base        = golapack.Base
	Digits      = golapack.Digits
	Round       = golapack.Round
	MinExponent = golapack.MinExponent
	MaxExponent = golapack.MaxExponent
)

var opts = mat.NewMatOptsCol()
var cmf = mat.CMatrixFactory()
var cvf = mat.CVectorFactory()
var mf = mat.MatrixFactory()
var vf = mat.VectorFactory()

func absint(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func maxf64(a ...float64) float64 {
	maxval := a[0]
	for _, val := range a {
		if val > maxval {
			maxval = val
		}
	}
	return maxval
}

func maxint(a ...int) int {
	maxval := a[0]
	for _, val := range a {
		if val > maxval {
			maxval = val
		}
	}
	return maxval
}

func maxintslice(a []int) int {
	var out int

	for _, val := range a {
		if val > out {
			out = val
		}
	}
	return out
}

func maxlocf64(a ...float64) int {
	maxval := a[0]
	maxloc := 1
	for i, val := range a {
		if val > maxval {
			maxval = val
			maxloc = i
		}
	}
	return maxloc + 1
}

func minf64(a ...float64) float64 {
	minval := a[0]
	for _, val := range a {
		if val < minval {
			minval = val
		}
	}
	return minval
}

func minint(a ...int) int {
	minval := a[0]
	for _, val := range a {
		if val < minval {
			minval = val
		}
	}
	return minval
}

func toSlice(x *[]int, n int) *[]int {
	y := (*x)[n:]
	return &y
}

func toSlicef64(x *[]float64, n int) *[]float64 {
	y := (*x)[n:]
	return &y
}

func toPtr(n int) *int {
	y := n
	return &y
}

func toPtrByte(n byte) *byte {
	y := n
	return &y
}

func toPtrf64(n float64) *float64 {
	y := n
	return &y
}

func toPtrc128(n complex128) *complex128 {
	y := n
	return &y
}

func toCmplx(n float64) complex128 {
	return complex(n, 0)
}

func genIter(a, b, inc int) []int {
	iter := make([]int, 0)

	if inc < 0 {
		for i := a; i >= b; i += inc {
			iter = append(iter, i)
		}
	} else {
		for i := a; i <= b; i += inc {
			iter = append(iter, i)
		}
	}
	return iter
}
