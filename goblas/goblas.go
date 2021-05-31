package goblas

import (
	"fmt"
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
		t.Errorf(" %6s: Incorrect number of tests performed: want %v got %v\n", name, x, n)
	} else {
		fmt.Printf(" %6s passed %6d computational tests\n", name, n)
	}
}
