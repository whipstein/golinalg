package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgetrs solves a system of linear equations
//    A * X = B,  A**T * X = B,  or  A**H * X = B
// with a general N-by-N matrix A using the LU factorization computed
// by ZGETRF.
func Zgetrs(trans mat.MatTrans, n, nrhs int, a *mat.CMatrix, ipiv *[]int, b *mat.CMatrix) (err error) {
	var notran bool
	var one complex128

	one = (1.0 + 0.0*1i)

	//     Test the input parameters.
	notran = trans == NoTrans
	if !notran && trans != Trans && trans != ConjTrans {
		err = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=%s", trans)
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
		gltest.Xerbla2("Zgetrs", err)
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
		Zlaswp(nrhs, b, 1, n, ipiv, 1)

		//        Solve L*X = B, overwriting B with X.
		if err = b.Trsm(Left, Lower, NoTrans, Unit, n, nrhs, one, a); err != nil {
			panic(err)
		}

		//        Solve U*X = B, overwriting B with X.
		if err = b.Trsm(Left, Upper, NoTrans, NonUnit, n, nrhs, one, a); err != nil {
			panic(err)
		}
	} else {
		//        Solve A**T * X = B  or A**H * X = B.
		//
		//        Solve U**T *X = B or U**H *X = B, overwriting B with X.
		if err = b.Trsm(Left, Upper, trans, NonUnit, n, nrhs, one, a); err != nil {
			panic(err)
		}

		//        Solve L**T *X = B, or L**H *X = B overwriting B with X.
		if err = b.Trsm(Left, Lower, trans, Unit, n, nrhs, one, a); err != nil {
			panic(err)
		}

		//        Apply row interchanges to the solution vectors.
		Zlaswp(nrhs, b, 1, n, ipiv, -1)
	}

	return
}
