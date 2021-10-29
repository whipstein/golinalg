package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zppcon estimates the reciprocal of the condition number (in the
// 1-norm) of a complex Hermitian positive definite packed matrix using
// the Cholesky factorization A = U**H*U or A = L*L**H computed by
// ZPPTRF.
//
// An estimate is obtained for norm(inv(A)), and the reciprocal of the
// condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
func Zppcon(uplo mat.MatUplo, n int, ap *mat.CVector, anorm float64, work *mat.CVector, rwork *mat.Vector) (rcond float64, err error) {
	var upper bool
	var normin byte
	var ainvnm, one, scale, scalel, scaleu, smlnum, zero float64
	var ix, kase int

	isave := make([]int, 3)

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if anorm < zero {
		err = fmt.Errorf("anorm < zero: anorm=%v", anorm)
	}
	if err != nil {
		gltest.Xerbla2("Zppcon", err)
		return
	}

	//     Quick return if possible
	rcond = zero
	if n == 0 {
		rcond = one
		return
	} else if anorm == zero {
		return
	}

	smlnum = Dlamch(SafeMinimum)

	//     Estimate the 1-norm of the inverse.
	kase = 0
	normin = 'N'
label10:
	;
	ainvnm, kase = Zlacn2(n, work.Off(n), work, ainvnm, kase, &isave)
	if kase != 0 {
		if upper {
			//           Multiply by inv(U**H).
			if scalel, err = Zlatps(Upper, ConjTrans, NonUnit, normin, n, ap, work, rwork); err != nil {
				panic(err)
			}
			normin = 'Y'

			//           Multiply by inv(U).
			if scaleu, err = Zlatps(Upper, NoTrans, NonUnit, normin, n, ap, work, rwork); err != nil {
				panic(err)
			}
		} else {
			//           Multiply by inv(L).
			if scalel, err = Zlatps(Lower, NoTrans, NonUnit, normin, n, ap, work, rwork); err != nil {
				panic(err)
			}
			normin = 'Y'

			//           Multiply by inv(L**H).
			if scaleu, err = Zlatps(Lower, ConjTrans, NonUnit, normin, n, ap, work, rwork); err != nil {
				panic(err)
			}
		}

		//        Multiply by 1/SCALE if doing so will not cause overflow.
		scale = scalel * scaleu
		if scale != one {
			ix = goblas.Izamax(n, work.Off(0, 1))
			if scale < cabs1(work.Get(ix-1))*smlnum || scale == zero {
				return
			}
			Zdrscl(n, scale, work.Off(0, 1))
		}
		goto label10
	}

	//     Compute the estimate of the reciprocal condition number.
	if ainvnm != zero {
		rcond = (one / ainvnm) / anorm
	}

	return
}
