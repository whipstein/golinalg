package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

type memory struct {
	infoc struct {
		infot int
		nunit int
		ok    bool
		lerr  bool
	}
	srnamc struct {
		srnamt string
	}
	sslct struct {
		selopt int
		seldim int
		selval []bool
		selwr  string
		selwi  string
	}
	claenv struct {
		iparms []int
	}
	cenvir struct {
		nproc  int
		nshift int
		maxb   int
	}
	mn struct {
		m      int
		n      int
		mplusn int
		k      int
		i      int
		fs     bool
	}
}

var common memory

var opts = mat.NewMatOptsCol()
var cmf = mat.CMatrixFactory()
var cmdf = mat.CMatrixDataFactory()
var cvf = mat.CVectorFactory()
var cvdf = mat.CVectorDataFactory()
var mf = mat.MatrixFactory()
var mdf = mat.MatrixDataFactory()
var vf = mat.VectorFactory()
var vdf = mat.VectorDataFactory()

func cabs1(cdum complex128) float64 {
	return math.Abs(real(cdum)) + math.Abs(imag(cdum))
}

func abs1(cdum complex128) float64 {
	return maxf64(math.Abs(real(cdum)) + math.Abs(imag(cdum)))
}

func abssq(cdum complex128) float64 {
	return math.Pow(real(cdum), 2) + math.Pow(imag(cdum), 2)
}

type Solution int

const (
	NewSolution Solution = iota
	Computed
)

type Norm int

const (
	OneNorm Norm = iota
	InfNorm
	MaxNorm
	FroNorm
)

func (n Norm) String() string {
	switch n {
	case OneNorm:
		return "OneNorm"
	case InfNorm:
		return "InfNorm"
	case MaxNorm:
		return "MaxNorm"
	case FroNorm:
		return "FroNorm"
	default:
		return ""
	}
}

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

type dslectFunc func(*float64, *float64) bool
type dlctesFunc func(*float64, *float64, *float64) bool

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

func toCmplxPtr(n float64) *complex128 {
	y := complex(n, 0)
	return &y
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
