package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpotrs solves a system of linear equations A*X = B with a symmetric
// positive definite matrix A using the Cholesky factorization
// A = U**T*U or A = L*L**T computed by DPOTRF.
func Dpotrs(uplo mat.MatUplo, n, nrhs int, a, b *mat.Matrix) (err error) {
	var upper bool

	one := 1.0

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
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
		gltest.Xerbla2("Dpotrs", err)
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 {
		return
	}

	if upper {
		//        Solve A*X = B where A = U**T *U.
		//
		//        Solve U**T *X = B, overwriting B with X.
		if err = b.Trsm(mat.Left, mat.Upper, mat.Trans, mat.NonUnit, n, nrhs, one, a); err != nil {
			panic(err)
		}

		//        Solve U*X = B, overwriting B with X.
		if err = b.Trsm(mat.Left, mat.Upper, mat.NoTrans, mat.NonUnit, n, nrhs, one, a); err != nil {
			panic(err)
		}
	} else {
		//        Solve A*X = B where A = L*L**T.
		//
		//        Solve L*X = B, overwriting B with X.
		if err = b.Trsm(mat.Left, mat.Lower, mat.NoTrans, mat.NonUnit, n, nrhs, one, a); err != nil {
			panic(err)
		}

		//        Solve L**T *X = B, overwriting B with X.
		if err = b.Trsm(mat.Left, mat.Lower, mat.Trans, mat.NonUnit, n, nrhs, one, a); err != nil {
			panic(err)
		}
	}

	return
}
