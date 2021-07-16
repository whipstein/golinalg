package goblas

import (
	"fmt"
	"math"
	"testing"

	"github.com/whipstein/golinalg/mat"
)

var opts = mat.NewMatOptsCol()
var cmf = mat.CMatrixFactory()
var cmdf = mat.CMatrixDataFactory()
var cvf = mat.CVectorFactory()
var cvdf = mat.CVectorDataFactory()
var mf = mat.MatrixFactory()
var mdf = mat.MatrixDataFactory()
var vf = mat.VectorFactory()
var vdf = mat.VectorDataFactory()

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
	blocksize    = 64
	minParBlocks = 4
)

func toPtr(n int) *int {
	y := n
	return &y
}

func toPtrf64(n float64) *float64 {
	y := n
	return &y
}

type SrotMatrix struct {
	flag   int
	matrix [4]float32
}

// type DrotMatrix struct {
// 	flag               int
// 	h11, h12, h21, h22 float64
// }

func passL1() {
	fmt.Printf(" %6s passed %6d computational tests\n", common.combla._case, common.combla.n)
}

func passL2(name string, n, x int, t *testing.T) {
	if n != x {
		t.Fail()
		fmt.Printf(" %6s: Incorrect number of tests performed: want %v got %v\n", name, x, n)
	} else {
		fmt.Printf(" %6s passed %6d computational tests\n", name, n)
	}
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

func genIter2(n, inc int) []int {
	iter := make([]int, 0)

	if inc > 0 {
		for i := 0; i < n; i++ {
			iter = append(iter, i*inc)
		}
	} else {
		for i := 0; i < n; i++ {
			iter = append(iter, (-n+1+i)*inc)
		}
	}
	return iter
}

// blocks returns the number of divisions of the dimension length with the given
// block size.
func blocks(dim int, bsize ...int) int {
	size := blocksize
	if len(bsize) > 0 {
		size = bsize[0]
	}
	return (dim + size - 1) / size
}

func abs1(c complex128) float64 {
	return math.Abs(real(c)) + math.Abs(imag(c))
}
