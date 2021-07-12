package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// DPBSV computes the solution to a real system of linear equations
//    A * X = B,
// where A is an N-by-N symmetric positive definite band matrix and X
// and B are N-by-NRHS matrices.
//
// The Cholesky decomposition is used to factor A as
//    A = U**T * U,  if UPLO = 'U', or
//    A = L * L**T,  if UPLO = 'L',
// where U is an upper triangular band matrix, and L is a lower
// triangular band matrix, with the same number of superdiagonals or
// subdiagonals as A.  The factored form of A is then used to solve the
// system of equations A * X = B.
func Dpbsv(uplo byte, n, kd, nrhs *int, ab *mat.Matrix, ldab *int, b *mat.Matrix, ldb, info *int) {
	//     Test the input parameters.
	(*info) = 0
	if uplo != 'U' && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*kd) < 0 {
		(*info) = -3
	} else if (*nrhs) < 0 {
		(*info) = -4
	} else if (*ldab) < (*kd)+1 {
		(*info) = -6
	} else if (*ldb) < max(1, *n) {
		(*info) = -8
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DPBSV "), -(*info))
		return
	}

	//     Compute the Cholesky factorization A = U**T*U or A = L*L**T.
	Dpbtrf(uplo, n, kd, ab, ldab, info)
	if (*info) == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		Dpbtrs(uplo, n, kd, nrhs, ab, ldab, b, ldb, info)

	}
}
