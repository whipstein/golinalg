package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zsysvrook computes the solution to a complex system of linear
// equations
//    A * X = B,
// where A is an N-by-N symmetric matrix and X and B are N-by-NRHS
// matrices.
//
// The diagonal pivoting method is used to factor A as
//    A = U * D * U**T,  if UPLO = 'U', or
//    A = L * D * L**T,  if UPLO = 'L',
// where U (or L) is a product of permutation and unit upper (lower)
// triangular matrices, and D is symmetric and block diagonal with
// 1-by-1 and 2-by-2 diagonal blocks.
//
// ZSYTRF_ROOK is called to compute the factorization of a complex
// symmetric matrix A using the bounded Bunch-Kaufman ("rook") diagonal
// pivoting method.
//
// The factored form of A is then used to solve the system
// of equations A * X = B by calling ZSYTRS_ROOK.
func Zsysvrook(uplo byte, n, nrhs *int, a *mat.CMatrix, lda *int, ipiv *[]int, b *mat.CMatrix, ldb *int, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var lwkopt int

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
	} else if (*lwork) < 1 && !lquery {
		(*info) = -10
	}

	if (*info) == 0 {
		if (*n) == 0 {
			lwkopt = 1
		} else {
			Zsytrfrook(uplo, n, a, lda, ipiv, work, toPtr(-1), info)
			lwkopt = int(work.GetRe(0))
		}
		work.SetRe(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZSYSV_ROOK"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Compute the factorization A = U*D*U**T or A = L*D*L**T.
	Zsytrfrook(uplo, n, a, lda, ipiv, work, lwork, info)
	if (*info) == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		//
		//        Solve with TRS_ROOK ( Use Level 2 BLAS)
		Zsytrsrook(uplo, n, nrhs, a, lda, ipiv, b, ldb, info)

	}

	work.SetRe(0, float64(lwkopt))
}
