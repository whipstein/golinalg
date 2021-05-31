package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// DPPSV computes the solution to a real system of linear equations
//    A * X = B,
// where A is an N-by-N symmetric positive definite matrix stored in
// packed format and X and B are N-by-NRHS matrices.
//
// The Cholesky decomposition is used to factor A as
//    A = U**T* U,  if UPLO = 'U', or
//    A = L * L**T,  if UPLO = 'L',
// where U is an upper triangular matrix and L is a lower triangular
// matrix.  The factored form of A is then used to solve the system of
// equations A * X = B.
func Dppsv(uplo byte, n, nrhs *int, ap *mat.Vector, b *mat.Matrix, ldb, info *int) {
	//     Test the input parameters.
	(*info) = 0
	if uplo != 'U' && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -6
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DPPSV "), -(*info))
		return
	}

	//     Compute the Cholesky factorization A = U**T*U or A = L*L**T.
	Dpptrf(uplo, n, ap, info)
	if (*info) == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		Dpptrs(uplo, n, nrhs, ap, b, ldb, info)

	}
}
