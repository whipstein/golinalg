package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
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
func Zpbsv(uplo byte, n, kd, nrhs *int, ab *mat.CMatrix, ldab *int, b *mat.CMatrix, ldb, info *int) {
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
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -8
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZPBSV "), -(*info))
		return
	}

	//     Compute the Cholesky factorization A = U**H *U or A = L*L**H.
	Zpbtrf(uplo, n, kd, ab, ldab, info)
	if (*info) == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		Zpbtrs(uplo, n, kd, nrhs, ab, ldab, b, ldb, info)

	}
}
