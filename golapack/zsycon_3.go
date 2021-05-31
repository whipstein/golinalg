package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsycon3 estimates the reciprocal of the condition number (in the
// 1-norm) of a complex symmetric matrix A using the factorization
// computed by ZSYTRF_RK or ZSYTRF_BK:
//
//    A = P*U*D*(U**T)*(P**T) or A = P*L*D*(L**T)*(P**T),
//
// where U (or L) is unit upper (or lower) triangular matrix,
// U**T (or L**T) is the transpose of U (or L), P is a permutation
// matrix, P**T is the transpose of P, and D is symmetric and block
// diagonal with 1-by-1 and 2-by-2 diagonal blocks.
//
// An estimate is obtained for norm(inv(A)), and the reciprocal of the
// condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
// This routine uses BLAS3 solver ZSYTRS_3.
func Zsycon3(uplo byte, n *int, a *mat.CMatrix, lda *int, e *mat.CVector, ipiv *[]int, anorm, rcond *float64, work *mat.CVector, info *int) {
	var upper bool
	var czero complex128
	var ainvnm, one, zero float64
	var i, kase int
	isave := make([]int, 3)

	one = 1.0
	zero = 0.0
	czero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *n) {
		(*info) = -4
	} else if (*anorm) < zero {
		(*info) = -7
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZSYCON_3"), -(*info))
		return
	}

	//     Quick return if possible
	(*rcond) = zero
	if (*n) == 0 {
		(*rcond) = one
		return
	} else if (*anorm) <= zero {
		return
	}

	//     Check that the diagonal matrix D is nonsingular.
	if upper {
		//        Upper triangular storage: examine D from bottom to top
		for i = (*n); i >= 1; i-- {
			if (*ipiv)[i-1] > 0 && a.Get(i-1, i-1) == czero {
				return
			}
		}
	} else {
		//        Lower triangular storage: examine D from top to bottom.
		for i = 1; i <= (*n); i++ {
			if (*ipiv)[i-1] > 0 && a.Get(i-1, i-1) == czero {
				return
			}
		}
	}

	//     Estimate the 1-norm of the inverse.
	kase = 0
label30:
	;
	Zlacn2(n, work.Off((*n)+1-1), work, &ainvnm, &kase, &isave)
	if kase != 0 {
		//        Multiply by inv(L*D*L**T) or inv(U*D*U**T).
		Zsytrs3(uplo, n, func() *int { y := 1; return &y }(), a, lda, e, ipiv, work.CMatrix(*n, opts), n, info)
		goto label30
	}

	//     Compute the estimate of the reciprocal condition number.
	if ainvnm != zero {
		(*rcond) = (one / ainvnm) / (*anorm)
	}
}
