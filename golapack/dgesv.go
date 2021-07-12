package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// DGESV computes the solution to a real system of linear equations
//    A * X = B,
// where A is an N-by-N matrix and X and B are N-by-NRHS matrices.
//
// The LU decomposition with partial pivoting and row interchanges is
// used to factor A as
//    A = P * L * U,
// where P is a permutation matrix, L is unit lower triangular, and U is
// upper triangular.  The factored form of A is then used to solve the
// system of equations A * X = B.
func Dgesv(n, nrhs *int, a *mat.Matrix, lda *int, ipiv *[]int, b *mat.Matrix, ldb *int, info *int) {
	(*info) = 0
	if (*n) < 0 {
		(*info) = -1
	} else if (*nrhs) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *n) {
		(*info) = -4
	} else if (*ldb) < max(1, *n) {
		(*info) = -7
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGESV "), -(*info))
		return
	}

	//     Compute the LU factorization of A.
	Dgetrf(n, n, a, lda, ipiv, info)
	if (*info) == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		Dgetrs('N', n, nrhs, a, lda, ipiv, b, ldb, info)
	}
}
