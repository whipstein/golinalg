package golapack

import (
	"fmt"

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
func Zgbcon(norm byte, n, kl, ku int, ab *mat.CMatrix, ipiv *[]int, anorm float64, work *mat.CVector, rwork *mat.Vector) (rcond float64, err error) {
	var lnoti, onenrm bool
	var normin byte
	var t complex128
	var ainvnm, one, scale, smlnum, zero float64
	var ix, j, jp, kase, kase1, kd, lm int

	isave := make([]int, 3)

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	onenrm = norm == '1' || norm == 'O'
	if !onenrm && norm != 'I' {
		err = fmt.Errorf("!onenrm && norm != 'I': norm='%c'", norm)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if kl < 0 {
		err = fmt.Errorf("kl < 0: kl=%v", kl)
	} else if ku < 0 {
		err = fmt.Errorf("ku < 0: ku=%v", ku)
	} else if ab.Rows < 2*kl+ku+1 {
		err = fmt.Errorf("ab.Rows < 2*kl+ku+1: ab.Rows=%v, kl=%v, ku=%v", ab.Rows, kl, ku)
	} else if anorm < zero {
		err = fmt.Errorf("anorm < zero: anorm=%v", anorm)
	}
	if err != nil {
		gltest.Xerbla2("Zgbcon", err)
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
	kd = kl + ku + 1
	lnoti = kl > 0
	kase = 0
label10:
	;
	ainvnm, kase = Zlacn2(n, work.Off(n), work, ainvnm, kase, &isave)
	if kase != 0 {
		if kase == kase1 {
			//           Multiply by inv(L).
			if lnoti {
				for j = 1; j <= n-1; j++ {
					lm = min(kl, n-j)
					jp = (*ipiv)[j-1]
					t = work.Get(jp - 1)
					if jp != j {
						work.Set(jp-1, work.Get(j-1))
						work.Set(j-1, t)
					}
					work.Off(j).Axpy(lm, -t, ab.Off(kd, j-1).CVector(), 1, 1)
				}
			}

			//           Multiply by inv(U).
			if scale, err = Zlatbs(Upper, NoTrans, NonUnit, normin, n, kl+ku, ab, work, rwork); err != nil {
				panic(err)
			}
		} else {
			//           Multiply by inv(U**H).
			if scale, err = Zlatbs(Upper, ConjTrans, NonUnit, normin, n, kl+ku, ab, work, rwork); err != nil {
				panic(err)
			}

			//           Multiply by inv(L**H).
			if lnoti {
				for j = n - 1; j >= 1; j-- {
					lm = min(kl, n-j)
					work.Set(j-1, work.Get(j-1)-work.Off(j).Dotc(lm, ab.Off(kd, j-1).CVector(), 1, 1))
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
			ix = work.Iamax(n, 1)
			if scale < cabs1(work.Get(ix-1))*smlnum || scale == zero {
				return
			}
			Zdrscl(n, scale, work, 1)
		}
		goto label10
	}

	//     Compute the estimate of the reciprocal condition number.
	if ainvnm != zero {
		rcond = (one / ainvnm) / anorm
	}

	return
}
