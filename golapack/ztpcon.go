package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Ztpcon estimates the reciprocal of the condition number of a packed
// triangular matrix A, in either the 1-norm or the infinity-norm.
//
// The norm of A is computed and an estimate is obtained for
// norm(inv(A)), then the reciprocal of the condition number is
// computed as
//    RCOND = 1 / ( norm(A) * norm(inv(A)) ).
func Ztpcon(norm, uplo, diag byte, n *int, ap *mat.CVector, rcond *float64, work *mat.CVector, rwork *mat.Vector, info *int) {
	var nounit, onenrm, upper bool
	var normin byte
	var ainvnm, anorm, one, scale, smlnum, xnorm, zero float64
	var ix, kase, kase1 int
	isave := make([]int, 3)

	one = 1.0
	zero = 0.0

	Cabs1 := func(zdum complex128) float64 { return math.Abs(real(zdum)) + math.Abs(imag(zdum)) }

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	onenrm = norm == '1' || norm == 'O'
	nounit = diag == 'N'

	if !onenrm && norm != 'I' {
		(*info) = -1
	} else if !upper && uplo != 'L' {
		(*info) = -2
	} else if !nounit && diag != 'U' {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTPCON"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		(*rcond) = one
		return
	}

	(*rcond) = zero
	smlnum = Dlamch(SafeMinimum) * float64(maxint(1, *n))

	//     Compute the norm of the triangular matrix A.
	anorm = Zlantp(norm, uplo, diag, n, ap, rwork)

	//     Continue only if ANORM > 0.
	if anorm > zero {
		//        Estimate the norm of the inverse of A.
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
				//              Multiply by inv(A).
				Zlatps(uplo, 'N', diag, normin, n, ap, work, &scale, rwork, info)
			} else {
				//              Multiply by inv(A**H).
				Zlatps(uplo, 'C', diag, normin, n, ap, work, &scale, rwork, info)
			}
			normin = 'Y'

			//           Multiply by 1/SCALE if doing so will not cause overflow.
			if scale != one {
				ix = goblas.Izamax(n, work, func() *int { y := 1; return &y }())
				xnorm = Cabs1(work.Get(ix - 1))
				if scale < xnorm*smlnum || scale == zero {
					return
				}
				Zdrscl(n, &scale, work, func() *int { y := 1; return &y }())
			}
			goto label10
		}

		//        Compute the estimate of the reciprocal condition number.
		if ainvnm != zero {
			(*rcond) = (one / anorm) / ainvnm
		}
	}
}