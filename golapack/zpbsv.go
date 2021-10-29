package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zpbsv computes the solution to a complex system of linear equations
//    A * X = B,
// where A is an N-by-N Hermitian positive definite band matrix and X
// and B are N-by-NRHS matrices.
//
// The Cholesky decomposition is used to factor A as
//    A = U**H * U,  if UPLO = 'U', or
//    A = L * L**H,  if UPLO = 'L',
// where U is an upper triangular band matrix, and L is a lower
// triangular band matrix, with the same number of superdiagonals or
// subdiagonals as A.  The factored form of A is then used to solve the
// system of equations A * X = B.
func Zpbsv(uplo mat.MatUplo, n, kd, nrhs int, ab, b *mat.CMatrix) (info int, err error) {
	//     Test the input parameters.
	if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if kd < 0 {
		err = fmt.Errorf("kd < 0: kd=%v", kd)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if ab.Rows < kd+1 {
		err = fmt.Errorf("ab.Rows < kd+1: ab.Rows=%v, kd=%v", ab.Rows, kd)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zpbsv", err)
		return
	}

	//     Compute the Cholesky factorization A = U**H *U or A = L*L**H.
	if info, err = Zpbtrf(uplo, n, kd, ab); err != nil {
		panic(err)
	}
	if info == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		if err = Zpbtrs(uplo, n, kd, nrhs, ab, b); err != nil {
			panic(err)
		}

	}

	return
}
