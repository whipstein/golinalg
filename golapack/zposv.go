package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
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
func Zposv(uplo byte, n, nrhs *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb, info *int) {
	//     Test the input parameters.
	(*info) = 0
	if uplo != 'U' && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -7
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZPOSV "), -(*info))
		return
	}

	//     Compute the Cholesky factorization A = U**H *U or A = L*L**H.
	Zpotrf(uplo, n, a, lda, info)
	if (*info) == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		Zpotrs(uplo, n, nrhs, a, lda, b, ldb, info)

	}
}
