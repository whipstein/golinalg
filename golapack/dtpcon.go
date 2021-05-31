package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtpcon estimates the reciprocal of the condition number of a packed
// triangular matrix A, in either the 1-norm or the infinity-norm.
//
// The norm of A is computed and an estimate is obtained for
// norm(inv(A)), then the reciprocal of the condition number is
// computed as
//    RCOND = 1 / ( norm(A) * norm(inv(A)) ).
func Dtpcon(norm, uplo, diag byte, n *int, ap *mat.Vector, rcond *float64, work *mat.Vector, iwork *[]int, info *int) {
	var nounit, onenrm, upper bool
	var normin byte
	var ainvnm, anorm, one, scale, smlnum, xnorm, zero float64
	var ix, kase, kase1 int
	isave := make([]int, 3)

	one = 1.0
	zero = 0.0

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
		gltest.Xerbla([]byte("DTPCON"), -(*info))
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
	anorm = Dlantp(norm, uplo, diag, n, ap, work)

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
		Dlacn2(n, work.Off((*n)+1-1), work, iwork, &ainvnm, &kase, &isave)
		if kase != 0 {
			if kase == kase1 {
				//              Multiply by inv(A).
				Dlatps(uplo, 'N', diag, normin, n, ap, work, &scale, work.Off(2*(*n)+1-1), info)
			} else {
				//              Multiply by inv(A**T).
				Dlatps(uplo, 'T', diag, normin, n, ap, work, &scale, work.Off(2*(*n)+1-1), info)
			}
			normin = 'Y'

			//           Multiply by 1/SCALE if doing so will not cause overflow.
			if scale != one {
				ix = goblas.Idamax(n, work, toPtr(1))
				xnorm = math.Abs(work.Get(ix - 1))
				if scale < xnorm*smlnum || scale == zero {
					return
				}
				Drscl(n, &scale, work, func() *int { y := 1; return &y }())
			}
			goto label10
		}

		//        Compute the estimate of the reciprocal condition number.
		if ainvnm != zero {
			(*rcond) = (one / anorm) / ainvnm
		}
	}
}
