package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zpocon estimates the reciprocal of the condition number (in the
// 1-norm) of a complex Hermitian positive definite matrix using the
// Cholesky factorization A = U**H*U or A = L*L**H computed by ZPOTRF.
//
// An estimate is obtained for norm(inv(A)), and the reciprocal of the
// condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
func Zpocon(uplo byte, n *int, a *mat.CMatrix, lda *int, anorm, rcond *float64, work *mat.CVector, rwork *mat.Vector, info *int) {
	var upper bool
	var normin byte
	var ainvnm, one, scale, scalel, scaleu, smlnum, zero float64
	var ix, kase int
	isave := make([]int, 3)

	one = 1.0
	zero = 0.0

	Cabs1 := func(zdum complex128) float64 { return math.Abs(real(zdum)) + math.Abs(imag(zdum)) }

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
		(*info) = -5
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZPOCON"), -(*info))
		return
	}

	//     Quick return if possible
	(*rcond) = zero
	if (*n) == 0 {
		(*rcond) = one
		return
	} else if (*anorm) == zero {
		return
	}

	smlnum = Dlamch(SafeMinimum)

	//     Estimate the 1-norm of inv(A).
	kase = 0
	normin = 'N'
label10:
	;
	Zlacn2(n, work.Off((*n)+1-1), work, &ainvnm, &kase, &isave)
	if kase != 0 {
		if upper {
			//           Multiply by inv(U**H).
			Zlatrs('U', 'C', 'N', normin, n, a, lda, work, &scalel, rwork, info)
			normin = 'Y'

			//           Multiply by inv(U).
			Zlatrs('U', 'N', 'N', normin, n, a, lda, work, &scaleu, rwork, info)
		} else {
			//           Multiply by inv(L).
			Zlatrs('L', 'N', 'N', normin, n, a, lda, work, &scalel, rwork, info)
			normin = 'Y'

			//           Multiply by inv(L**H).
			Zlatrs('L', 'C', 'N', normin, n, a, lda, work, &scaleu, rwork, info)
		}

		//        Multiply by 1/SCALE if doing so will not cause overflow.
		scale = scalel * scaleu
		if scale != one {
			ix = goblas.Izamax(*n, work, 1)
			if scale < Cabs1(work.Get(ix-1))*smlnum || scale == zero {
				return
			}
			Zdrscl(n, &scale, work, func() *int { y := 1; return &y }())
		}
		goto label10
	}

	//     Compute the estimate of the reciprocal condition number.
	if ainvnm != zero {
		(*rcond) = (one / ainvnm) / (*anorm)
	}
}
