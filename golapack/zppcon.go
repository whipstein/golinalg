package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Zppcon estimates the reciprocal of the condition number (in the
// 1-norm) of a complex Hermitian positive definite packed matrix using
// the Cholesky factorization A = U**H*U or A = L*L**H computed by
// ZPPTRF.
//
// An estimate is obtained for norm(inv(A)), and the reciprocal of the
// condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
func Zppcon(uplo byte, n *int, ap *mat.CVector, anorm, rcond *float64, work *mat.CVector, rwork *mat.Vector, info *int) {
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
	} else if (*anorm) < zero {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZPPCON"), -(*info))
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
	Zlacn2(n, work.Off((*n)+1-1), work, &ainvnm, &kase, &isave)
	if kase != 0 {
		if upper {
			//           Multiply by inv(U**H).
			Zlatps('U', 'C', 'N', normin, n, ap, work, &scalel, rwork, info)
			normin = 'Y'

			//           Multiply by inv(U).
			Zlatps('U', 'N', 'N', normin, n, ap, work, &scaleu, rwork, info)
		} else {
			//           Multiply by inv(L).
			Zlatps('L', 'N', 'N', normin, n, ap, work, &scalel, rwork, info)
			normin = 'Y'

			//           Multiply by inv(L**H).
			Zlatps('L', 'C', 'N', normin, n, ap, work, &scaleu, rwork, info)
		}

		//        Multiply by 1/SCALE if doing so will not cause overflow.
		scale = scalel * scaleu
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
