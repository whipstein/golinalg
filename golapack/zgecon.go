package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Zgecon estimates the reciprocal of the condition number of a general
// complex matrix A, in either the 1-norm or the infinity-norm, using
// the LU factorization computed by ZGETRF.
//
// An estimate is obtained for norm(inv(A)), and the reciprocal of the
// condition number is computed as
//    RCOND = 1 / ( norm(A) * norm(inv(A)) ).
func Zgecon(norm byte, n *int, a *mat.CMatrix, lda *int, anorm, rcond *float64, work *mat.CVector, rwork *mat.Vector, info *int) {
	var onenrm bool
	var normin byte
	var ainvnm, one, scale, sl, smlnum, su, zero float64
	var ix, kase, kase1 int
	isave := make([]int, 3)

	one = 1.0
	zero = 0.0

	Cabs1 := func(zdum complex128) float64 { return math.Abs(real(zdum)) + math.Abs(imag(zdum)) }

	//     Test the input parameters.
	(*info) = 0
	onenrm = norm == '1' || norm == 'O'
	if !onenrm && norm != 'I' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *n) {
		(*info) = -4
	} else if (*anorm) < zero {
		(*info) = -5
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGECON"), -(*info))
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

	//     Estimate the norm of inv(A).
	ainvnm = zero
	normin = 'N'
	if onenrm {
		kase1 = 1
	} else {
		kase1 = 2
	}
	kase = 0
label10:
	;
	Zlacn2(n, work.Off((*n)+1-1), work, &ainvnm, &kase, &isave)
	if kase != 0 {
		if kase == kase1 {
			//           Multiply by inv(L).
			Zlatrs('L', 'N', 'U', normin, n, a, lda, work, &sl, rwork, info)

			//           Multiply by inv(U).
			Zlatrs('U', 'N', 'N', normin, n, a, lda, work, &su, rwork.Off((*n)+1-1), info)
		} else {
			//           Multiply by inv(U**H).
			Zlatrs('U', 'C', 'N', normin, n, a, lda, work, &su, rwork.Off((*n)+1-1), info)

			//           Multiply by inv(L**H).
			Zlatrs('L', 'C', 'U', normin, n, a, lda, work, &sl, rwork, info)
		}

		//        Divide X by 1/(SL*SU) if doing so will not cause overflow.
		scale = sl * su
		normin = 'Y'
		if scale != one {
			ix = goblas.Izamax(n, work, func() *int { y := 1; return &y }())
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
