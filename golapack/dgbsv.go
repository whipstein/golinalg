package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dgbsv computes the solution to a real system of linear equations
// A * X = B, where A is a band matrix of order N with KL subdiagonals
// and KU superdiagonals, and X and B are N-by-NRHS matrices.
//
// The LU decomposition with partial pivoting and row interchanges is
// used to factor A as A = L * U, where L is a product of permutation
// and unit lower triangular matrices with KL subdiagonals, and U is
// upper triangular with KL+KU superdiagonals.  The factored form of A
// is then used to solve the system of equations A * X = B.
func Dgbsv(n, kl, ku, nrhs *int, ab *mat.Matrix, ldab *int, ipiv *[]int, b *mat.Matrix, ldb, info *int) {
	//     Test the input parameters.
	(*info) = 0
	if (*n) < 0 {
		(*info) = -1
	} else if (*kl) < 0 {
		(*info) = -2
	} else if (*ku) < 0 {
		(*info) = -3
	} else if (*nrhs) < 0 {
		(*info) = -4
	} else if (*ldab) < 2*(*kl)+(*ku)+1 {
		(*info) = -6
	} else if (*ldb) < maxint(*n, 1) {
		(*info) = -9
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGBSV "), -(*info))
		return
	}

	//     Compute the LU factorization of the band matrix A.
	Dgbtrf(n, n, kl, ku, ab, ldab, ipiv, info)
	if (*info) == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		Dgbtrs('N', n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info)
	}
}
