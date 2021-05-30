package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zsysvaa computes the solution to a complex system of linear equations
//    A * X = B,
// where A is an N-by-N symmetric matrix and X and B are N-by-NRHS
// matrices.
//
// Aasen's algorithm is used to factor A as
//    A = U**T * T * U,  if UPLO = 'U', or
//    A = L * T * L**T,  if UPLO = 'L',
// where U (or L) is a product of permutation and unit upper (lower)
// triangular matrices, and T is symmetric tridiagonal. The factored
// form of A is then used to solve the system of equations A * X = B.
func Zsysvaa(uplo byte, n, nrhs *int, a *mat.CMatrix, lda *int, ipiv *[]int, b *mat.CMatrix, ldb *int, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var lwkopt, lwkoptSytrf, lwkoptSytrs int

	//     Test the input parameters.
	(*info) = 0
	lquery = ((*lwork) == -1)
	if uplo != 'U' && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -8
	} else if (*lwork) < maxint(2*(*n), 3*(*n)-2) && !lquery {
		(*info) = -10
	}
	//
	if (*info) == 0 {
		Zsytrfaa(uplo, n, a, lda, ipiv, work, toPtr(-1), info)
		lwkoptSytrf = int(work.GetRe(0))
		Zsytrsaa(uplo, n, nrhs, a, lda, ipiv, b, ldb, work, toPtr(-1), info)
		lwkoptSytrs = int(work.GetRe(0))
		lwkopt = maxint(lwkoptSytrf, lwkoptSytrs)
		work.SetRe(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZSYSV_AA "), -(*info))
		return
	} else if lquery {
		return
	}

	//     Compute the factorization A = U**T*T*U or A = L*T*L**T.
	Zsytrfaa(uplo, n, a, lda, ipiv, work, lwork, info)
	if (*info) == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		Zsytrsaa(uplo, n, nrhs, a, lda, ipiv, b, ldb, work, lwork, info)

	}

	work.SetRe(0, float64(lwkopt))
}
