package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgecon estimates the reciprocal of the condition number of a general
// real matrix A, in either the 1-norm or the infinity-norm, using
// the LU factorization computed by DGETRF.
//
// An estimate is obtained for norm(inv(A)), and the reciprocal of the
// condition number is computed as
//    RCOND = 1 / ( norm(A) * norm(inv(A)) ).
func Dgecon(norm byte, n *int, a *mat.Matrix, lda *int, anorm *float64, rcond *float64, work *mat.Vector, iwork *[]int, info *int) {
	var onenrm bool
	var normin byte
	var ainvnm, one, scale, sl, smlnum, su, zero float64
	var ix, kase, kase1 int
	isave := make([]int, 3)

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	(*info) = 0
	onenrm = norm == '1' || norm == 'O'
	if !onenrm && norm != 'I' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *n) {
		(*info) = -4
	} else if (*anorm) < zero {
		(*info) = -5
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGECON"), -(*info))
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
	//
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
	Dlacn2(n, work.Off((*n)), work, iwork, &ainvnm, &kase, &isave)
	if kase != 0 {
		if kase == kase1 {
			//           Multiply by inv(L).
			Dlatrs('L', 'N', 'U', normin, n, a, lda, work, &sl, work.Off(2*(*n)), info)

			//           Multiply by inv(U).
			Dlatrs('U', 'N', 'N', normin, n, a, lda, work, &su, work.Off(3*(*n)), info)
		} else {
			//           Multiply by inv(U**T).
			Dlatrs('U', 'T', 'N', normin, n, a, lda, work, &su, work.Off(3*(*n)), info)

			//           Multiply by inv(L**T).
			Dlatrs('L', 'T', 'U', normin, n, a, lda, work, &sl, work.Off(2*(*n)), info)
		}

		//        Divide X by 1/(SL*SU) if doing so will not cause overflow.
		scale = sl * su
		normin = 'Y'
		if scale != one {
			ix = goblas.Idamax(*n, work.Off(0, 1))
			if scale < math.Abs(work.Get(ix-1))*smlnum || scale == zero {
				goto label20
			}
			Drscl(n, &scale, work, func() *int { y := 1; return &y }())
		}
		goto label10
	}

	//     Compute the estimate of the reciprocal condition number.
	if ainvnm != zero {
		(*rcond) = (one / ainvnm) / (*anorm)
	}

label20:
}
