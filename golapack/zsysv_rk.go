package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsysvrk computes the solution to a complex system of linear
// equations A * X = B, where A is an N-by-N symmetric matrix
// and X and B are N-by-NRHS matrices.
//
// The bounded Bunch-Kaufman (rook) diagonal pivoting method is used
// to factor A as
//    A = P*U*D*(U**T)*(P**T),  if UPLO = 'U', or
//    A = P*L*D*(L**T)*(P**T),  if UPLO = 'L',
// where U (or L) is unit upper (or lower) triangular matrix,
// U**T (or L**T) is the transpose of U (or L), P is a permutation
// matrix, P**T is the transpose of P, and D is symmetric and block
// diagonal with 1-by-1 and 2-by-2 diagonal blocks.
//
// ZSYTRF_RK is called to compute the factorization of a complex
// symmetric matrix.  The factored form of A is then used to solve
// the system of equations A * X = B by calling BLAS3 routine ZSYTRS_3.
func Zsysvrk(uplo byte, n, nrhs *int, a *mat.CMatrix, lda *int, e *mat.CVector, ipiv *[]int, b *mat.CMatrix, ldb *int, work *mat.CVector, lwork, info *int) {
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
		(*info) = -9
	} else if (*lwork) < 1 && !lquery {
		(*info) = -11
	}

	if (*info) == 0 {
		if (*n) == 0 {
			lwkopt = 1
		} else {
			Zsytrfrk(uplo, n, a, lda, e, ipiv, work, toPtr(-1), info)
			lwkopt = int(work.GetRe(0))
		}
		work.SetRe(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZSYSV_RK"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Compute the factorization A = P*U*D*(U**T)*(P**T) or
	//     A = P*U*D*(U**T)*(P**T).
	Zsytrfrk(uplo, n, a, lda, e, ipiv, work, lwork, info)

	if (*info) == 0 {
		//        Solve the system A*X = B with BLAS3 solver, overwriting B with X.
		Zsytrs3(uplo, n, nrhs, a, lda, e, ipiv, b, ldb, info)

	}

	work.SetRe(0, float64(lwkopt))
}
