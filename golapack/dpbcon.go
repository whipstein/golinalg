package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpbcon estimates the reciprocal of the condition number (in the
// 1-norm) of a real symmetric positive definite band matrix using the
// Cholesky factorization A = U**T*U or A = L*L**T computed by DPBTRF.
//
// An estimate is obtained for norm(inv(A)), and the reciprocal of the
// condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
func Dpbcon(uplo byte, n, kd *int, ab *mat.Matrix, ldab *int, anorm, rcond *float64, work *mat.Vector, iwork *[]int, info *int) {
	var upper bool
	var normin byte
	var ainvnm, one, scale, scalel, scaleu, smlnum, zero float64
	var ix, kase int
	isave := make([]int, 3)

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*kd) < 0 {
		(*info) = -3
	} else if (*ldab) < (*kd)+1 {
		(*info) = -5
	} else if (*anorm) < zero {
		(*info) = -6
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DPBCON"), -(*info))
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

	//     Estimate the 1-norm of the inverse.
	kase = 0
	normin = 'N'
label10:
	;
	Dlacn2(n, work.Off((*n)+1-1), work, iwork, &ainvnm, &kase, &isave)
	if kase != 0 {
		if upper {
			//           Multiply by inv(U**T).
			Dlatbs('U', 'T', 'N', normin, n, kd, ab, ldab, work, &scalel, work.Off(2*(*n)+1-1), info)
			normin = 'Y'

			//           Multiply by inv(U).
			Dlatbs('U', 'N', 'N', normin, n, kd, ab, ldab, work, &scaleu, work.Off(2*(*n)+1-1), info)
		} else {
			//           Multiply by inv(L).
			Dlatbs('L', 'N', 'N', normin, n, kd, ab, ldab, work, &scalel, work.Off(2*(*n)+1-1), info)
			normin = 'Y'

			//           Multiply by inv(L**T).
			Dlatbs('L', 'T', 'N', normin, n, kd, ab, ldab, work, &scaleu, work.Off(2*(*n)+1-1), info)
		}

		//        Multiply by 1/SCALE if doing so will not cause overflow.
		scale = scalel * scaleu
		if scale != one {
			ix = goblas.Idamax(n, work, toPtr(1))
			if scale < math.Abs(work.Get(ix-1))*smlnum || scale == zero {
				return
			}
			Drscl(n, &scale, work, func() *int { y := 1; return &y }())
		}
		goto label10
	}

	//     Compute the estimate of the reciprocal condition number.
	if ainvnm != zero {
		(*rcond) = (one / ainvnm) / (*anorm)
	}
}
