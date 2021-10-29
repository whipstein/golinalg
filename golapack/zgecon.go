package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgecon estimates the reciprocal of the condition number of a general
// complex matrix A, in either the 1-norm or the infinity-norm, using
// the LU factorization computed by ZGETRF.
//
// An estimate is obtained for norm(inv(A)), and the reciprocal of the
// condition number is computed as
//    RCOND = 1 / ( norm(A) * norm(inv(A)) ).
func Zgecon(norm byte, n int, a *mat.CMatrix, anorm float64, work *mat.CVector, rwork *mat.Vector) (rcond float64, err error) {
	var onenrm bool
	var normin byte
	var ainvnm, one, scale, sl, smlnum, su, zero float64
	var ix, kase, kase1 int

	isave := make([]int, 3)

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	onenrm = norm == '1' || norm == 'O'
	if !onenrm && norm != 'I' {
		err = fmt.Errorf("!onenrm && norm != 'I': norm='%c'", norm)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if anorm < zero {
		err = fmt.Errorf("anorm < zero: anorm=%v", anorm)
	}
	if err != nil {
		gltest.Xerbla2("Zgecon", err)
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
	ainvnm, kase = Zlacn2(n, work.Off(n), work, ainvnm, kase, &isave)
	if kase != 0 {
		if kase == kase1 {
			//           Multiply by inv(L).
			if sl, err = Zlatrs(Lower, NoTrans, Unit, normin, n, a, work, rwork); err != nil {
				panic(err)
			}

			//           Multiply by inv(U).
			if su, err = Zlatrs(Upper, NoTrans, NonUnit, normin, n, a, work, rwork.Off(n)); err != nil {
				panic(err)
			}
		} else {
			//           Multiply by inv(U**H).
			if su, err = Zlatrs(Upper, ConjTrans, NonUnit, normin, n, a, work, rwork.Off(n)); err != nil {
				panic(err)
			}

			//           Multiply by inv(L**H).
			if sl, err = Zlatrs(Lower, ConjTrans, Unit, normin, n, a, work, rwork); err != nil {
				panic(err)
			}
		}

		//        Divide X by 1/(SL*SU) if doing so will not cause overflow.
		scale = sl * su
		normin = 'Y'
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
