package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgetrs solves a system of linear equations
//    A * X = B  or  A**T * X = B
// with a general N-by-N matrix A using the LU factorization computed
// by DGETRF.
func Dgetrs(trans mat.MatTrans, n, nrhs int, a *mat.Matrix, ipiv []int, b *mat.Matrix) (err error) {
	var notran bool

	one := 1.0

	//     Test the input parameters.
	notran = trans == NoTrans
	if !trans.IsValid() {
		err = fmt.Errorf("!trans.IsValid(): trans=%s", trans)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dgetrs", err)
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 {
		return
	}

	if notran {
		//        Solve A * X = B.
		//
		//        Apply row interchanges to the right hand sides.
		Dlaswp(nrhs, b, 1, n, ipiv, 1)

		//        Solve L*X = B, overwriting B with X.
		if err = goblas.Dtrsm(mat.Left, mat.Lower, mat.NoTrans, mat.Unit, n, nrhs, one, a, b); err != nil {
			panic(err)
		}

		//        Solve U*X = B, overwriting B with X.
		if err = goblas.Dtrsm(mat.Left, mat.Upper, mat.NoTrans, mat.NonUnit, n, nrhs, one, a, b); err != nil {
			panic(err)
		}
	} else {
		//        Solve A**T * X = B.
		//
		//        Solve U**T *X = B, overwriting B with X.
		if err = goblas.Dtrsm(mat.Left, mat.Upper, mat.Trans, mat.NonUnit, n, nrhs, one, a, b); err != nil {
			panic(err)
		}

		//        Solve L**T *X = B, overwriting B with X.
		if err = goblas.Dtrsm(mat.Left, mat.Lower, mat.Trans, mat.Unit, n, nrhs, one, a, b); err != nil {
			panic(err)
		}

		//        Apply row interchanges to the solution vectors.
		Dlaswp(nrhs, b, 1, n, ipiv, -1)
	}

	return
}
