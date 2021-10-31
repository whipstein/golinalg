package math

import (
	"math"
	"math/cmplx"
	"strconv"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

var cvf = mat.CVectorFactory()
var mf = mat.MatrixFactory()
var vf = mat.VectorFactory()
var opts = mat.NewMatOpts()

func MatInv(m *mat.Matrix) {
	var colMajor int

	work := mf(m.Rows, m.Cols)
	ipiv := make([]int, m.Rows)

	if m.Rows > 1 {
		// Calculate minv=inv(matrix)
		if m.Opts.Major == mat.Row {
			m.ToColMajor()
			colMajor = 1
		}
		if info, err := golapack.Dgetrf(m.Rows, m.Cols, m, &ipiv); err != nil || info != 0 {
			panic("golapack.Zgetrf error: " + strconv.Itoa(info))
		}
		if info, err := golapack.Dgetri(m.Rows, m, ipiv, work); err != nil || info != 0 {
			panic("golapack.Zgetri error: " + strconv.Itoa(info))
		}
		if colMajor == 1 {
			m.ToRowMajor()
			colMajor = 0
		}
	} else {
		m.Set(0, 0, math.Pow(m.Get(0, 0), -1))
	}
}

func CMatInv(m *mat.CMatrix) {
	var colMajor int

	work := cvf(m.Rows * m.Cols)
	lwork := m.Rows * m.Cols
	ipiv := make([]int, m.Rows)

	if m.Rows > 1 {
		// Calculate minv=inv(matrix)
		if m.Opts.Major == mat.Row {
			m.ToColMajor()
			colMajor = 1
		}
		if info, err := golapack.Zgetrf(m.Rows, m.Cols, m, &ipiv); err != nil || info != 0 {
			panic("golapack.Zgetrf error: " + strconv.Itoa(info))
		}
		if info, err := golapack.Zgetri(m.Rows, m, &ipiv, work, lwork); err != nil || info != 0 {
			panic("golapack.Zgetri error: " + strconv.Itoa(info))
		}
		if colMajor == 1 {
			m.ToRowMajor()
			colMajor = 0
		}
	} else {
		m.Set(0, 0, cmplx.Pow(m.Get(0, 0), -1))
	}
}
