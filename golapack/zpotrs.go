package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zpotrs solves a system of linear equations A*X = B with a Hermitian
// positive definite matrix A using the Cholesky factorization
// A = U**H * U or A = L * L**H computed by ZPOTRF.
func Zpotrs(uplo mat.MatUplo, n, nrhs int, a, b *mat.CMatrix) (err error) {
	var upper bool
	var one complex128

	one = (1.0 + 0.0*1i)

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
		gltest.Xerbla2("Zpotrs", err)
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 {
		return
	}

	if upper {
		//        Solve A*X = B where A = U**H *U.
		//
		//        Solve U**H *X = B, overwriting B with X.
		if err = b.Trsm(Left, Upper, ConjTrans, NonUnit, n, nrhs, one, a); err != nil {
			panic(err)
		}

		//        Solve U*X = B, overwriting B with X.
		if err = b.Trsm(Left, Upper, NoTrans, NonUnit, n, nrhs, one, a); err != nil {
			panic(err)
		}
	} else {
		//        Solve A*X = B where A = L*L**H.
		//
		//        Solve L*X = B, overwriting B with X.
		if err = b.Trsm(Left, Lower, NoTrans, NonUnit, n, nrhs, one, a); err != nil {
			panic(err)
		}

		//        Solve L**H *X = B, overwriting B with X.
		if err = b.Trsm(Left, Lower, ConjTrans, NonUnit, n, nrhs, one, a); err != nil {
			panic(err)
		}
	}

	return
}
