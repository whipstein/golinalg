package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// DPOSV computes the solution to a real system of linear equations
//    A * X = B,
// where A is an N-by-N symmetric positive definite matrix and X and B
// are N-by-NRHS matrices.
//
// The Cholesky decomposition is used to factor A as
//    A = U**T* U,  if UPLO = 'U', or
//    A = L * L**T,  if UPLO = 'L',
// where U is an upper triangular matrix and L is a lower triangular
// matrix.  The factored form of A is then used to solve the system of
// equations A * X = B.
func Dposv(uplo byte, n, nrhs *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb, info *int) {
	//     Test the input parameters.
	(*info) = 0
	if uplo != 'U' && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*lda) < max(1, *n) {
		(*info) = -5
	} else if (*ldb) < max(1, *n) {
		(*info) = -7
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DPOSV "), -(*info))
		return
	}

	//     Compute the Cholesky factorization A = U**T*U or A = L*L**T.
	Dpotrf(uplo, n, a, lda, info)
	if (*info) == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		Dpotrs(uplo, n, nrhs, a, lda, b, ldb, info)

	}
}
