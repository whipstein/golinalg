package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtbcon estimates the reciprocal of the condition number of a
// triangular band matrix A, in either the 1-norm or the infinity-norm.
//
// The norm of A is computed and an estimate is obtained for
// norm(inv(A)), then the reciprocal of the condition number is
// computed as
//    RCOND = 1 / ( norm(A) * norm(inv(A)) ).
func Dtbcon(norm byte, uplo mat.MatUplo, diag mat.MatDiag, n, kd int, ab *mat.Matrix, work *mat.Vector, iwork *[]int) (rcond float64, err error) {
	var nounit, onenrm, upper bool
	var normin byte
	var ainvnm, anorm, one, scale, smlnum, xnorm, zero float64
	var ix, kase, kase1 int

	isave := make([]int, 3)

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	upper = uplo == Upper
	onenrm = norm == '1' || norm == 'O'
	nounit = diag == NonUnit

	if !onenrm && norm != 'I' {
		err = fmt.Errorf("!onenrm && norm != 'I': norm='%c'", norm)
	} else if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if !nounit && diag != Unit {
		err = fmt.Errorf("!nounit && diag != Unit: diag=%s", diag)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if kd < 0 {
		err = fmt.Errorf("kd < 0: kd=%v", kd)
	} else if ab.Rows < kd+1 {
		err = fmt.Errorf("ab.Rows < kd+1: ab.Rows=%v, kd=%v", ab.Rows, kd)
	}
	if err != nil {
		gltest.Xerbla2("Dtbcon", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		rcond = one
		return
	}

	rcond = zero
	smlnum = Dlamch(SafeMinimum) * float64(max(1, n))

	//     Compute the norm of the triangular matrix A.
	anorm = Dlantb(norm, uplo, diag, n, kd, ab, work)

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
		ainvnm, kase = Dlacn2(n, work.Off(n), work, iwork, ainvnm, kase, &isave)
		if kase != 0 {
			if kase == kase1 {
				//              Multiply by inv(A).
				if scale, err = Dlatbs(uplo, NoTrans, diag, normin, n, kd, ab, work, work.Off(2*n)); err != nil {
					panic(err)
				}
			} else {
				//              Multiply by inv(A**T).
				if scale, err = Dlatbs(uplo, Trans, diag, normin, n, kd, ab, work, work.Off(2*n)); err != nil {
					panic(err)
				}
			}
			normin = 'Y'

			//           Multiply by 1/SCALE if doing so will not cause overflow.
			if scale != one {
				ix = work.Iamax(n, 1)
				xnorm = math.Abs(work.Get(ix - 1))
				if scale < xnorm*smlnum || scale == zero {
					return
				}
				Drscl(n, scale, work, 1)
			}
			goto label10
		}

		//        Compute the estimate of the reciprocal condition number.
		if ainvnm != zero {
			rcond = (one / anorm) / ainvnm
		}
	}

	return
}
