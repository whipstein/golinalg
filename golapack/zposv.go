package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zposv computes the solution to a complex system of linear equations
//    A * X = B,
// where A is an N-by-N Hermitian positive definite matrix and X and B
// are N-by-NRHS matrices.
//
// The Cholesky decomposition is used to factor A as
//    A = U**H* U,  if UPLO = 'U', or
//    A = L * L**H,  if UPLO = 'L',
// where U is an upper triangular matrix and  L is a lower triangular
// matrix.  The factored form of A is then used to solve the system of
// equations A * X = B.
func Zposv(uplo mat.MatUplo, n, nrhs int, a, b *mat.CMatrix) (info int, err error) {
	//     Test the input parameters.
	if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=%s", uplo)
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
		gltest.Xerbla2("Zposv", err)
		return
	}

	//     Compute the Cholesky factorization A = U**H *U or A = L*L**H.
	if info, err = Zpotrf(uplo, n, a); err != nil {
		panic(err)
	}
	if info == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		if err = Zpotrs(uplo, n, nrhs, a, b); err != nil {
			panic(err)
		}

	}

	return
}
