package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgbcon estimates the reciprocal of the condition number of a complex
// general band matrix A, in either the 1-norm or the infinity-norm,
// using the LU factorization computed by ZGBTRF.
//
// An estimate is obtained for norm(inv(A)), and the reciprocal of the
// condition number is computed as
//    RCOND = 1 / ( norm(A) * norm(inv(A)) ).
func Zgbcon(norm byte, n, kl, ku *int, ab *mat.CMatrix, ldab *int, ipiv *[]int, anorm, rcond *float64, work *mat.CVector, rwork *mat.Vector, info *int) {
	var lnoti, onenrm bool
	var normin byte
	var t complex128
	var ainvnm, one, scale, smlnum, zero float64
	var ix, j, jp, kase, kase1, kd, lm int
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
	} else if (*kl) < 0 {
		(*info) = -3
	} else if (*ku) < 0 {
		(*info) = -4
	} else if (*ldab) < 2*(*kl)+(*ku)+1 {
		(*info) = -6
	} else if (*anorm) < zero {
		(*info) = -8
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGBCON"), -(*info))
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
	kd = (*kl) + (*ku) + 1
	lnoti = (*kl) > 0
	kase = 0
label10:
	;
	Zlacn2(n, work.Off((*n)), work, &ainvnm, &kase, &isave)
	if kase != 0 {
		if kase == kase1 {
			//           Multiply by inv(L).
			if lnoti {
				for j = 1; j <= (*n)-1; j++ {
					lm = min(*kl, (*n)-j)
					jp = (*ipiv)[j-1]
					t = work.Get(jp - 1)
					if jp != j {
						work.Set(jp-1, work.Get(j-1))
						work.Set(j-1, t)
					}
					goblas.Zaxpy(lm, -t, ab.CVector(kd, j-1, 1), work.Off(j, 1))
				}
			}

			//           Multiply by inv(U).
			Zlatbs('U', 'N', 'N', normin, n, toPtr((*kl)+(*ku)), ab, ldab, work, &scale, rwork, info)
		} else {
			//           Multiply by inv(U**H).
			Zlatbs('U', 'C', 'N', normin, n, toPtr((*kl)+(*ku)), ab, ldab, work, &scale, rwork, info)

			//           Multiply by inv(L**H).
			if lnoti {
				for j = (*n) - 1; j >= 1; j-- {
					lm = min(*kl, (*n)-j)
					work.Set(j-1, work.Get(j-1)-goblas.Zdotc(lm, ab.CVector(kd, j-1, 1), work.Off(j, 1)))
					jp = (*ipiv)[j-1]
					if jp != j {
						t = work.Get(jp - 1)
						work.Set(jp-1, work.Get(j-1))
						work.Set(j-1, t)
					}
				}
			}
		}

		//        Divide X by 1/SCALE if doing so will not cause overflow.
		normin = 'Y'
		if scale != one {
			ix = goblas.Izamax(*n, work.Off(0, 1))
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
