package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpbcon estimates the reciprocal of the condition number (in the
// 1-norm) of a real symmetric positive definite band matrix using the
// Cholesky factorization A = U**T*U or A = L*L**T computed by DPBTRF.
//
// An estimate is obtained for norm(inv(A)), and the reciprocal of the
// condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
func Dpbcon(uplo mat.MatUplo, n, kd int, ab *mat.Matrix, anorm float64, work *mat.Vector, iwork *[]int) (rcond float64, err error) {
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
	} else if kd < 0 {
		err = fmt.Errorf("kd < 0: kd=%v", kd)
	} else if ab.Rows < kd+1 {
		err = fmt.Errorf("ab.Rows < kd+1: ab.Rows=%v, kd=%v", ab.Rows, kd)
	} else if anorm < zero {
		err = fmt.Errorf("anorm < zero: anorm=%v", anorm)
	}
	if err != nil {
		gltest.Xerbla2("Dpbcon", err)
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
	ainvnm, kase = Dlacn2(n, work.Off(n), work, iwork, ainvnm, kase, &isave)
	if kase != 0 {
		if upper {
			//           Multiply by inv(U**T).
			if scalel, err = Dlatbs(Upper, Trans, NonUnit, normin, n, kd, ab, work, work.Off(2*n)); err != nil {
				panic(err)
			}
			normin = 'Y'

			//           Multiply by inv(U).
			if scaleu, err = Dlatbs(Upper, NoTrans, NonUnit, normin, n, kd, ab, work, work.Off(2*n)); err != nil {
				panic(err)
			}
		} else {
			//           Multiply by inv(L).
			if scalel, err = Dlatbs(Lower, NoTrans, NonUnit, normin, n, kd, ab, work, work.Off(2*n)); err != nil {
				panic(err)
			}
			normin = 'Y'

			//           Multiply by inv(L**T).
			if scaleu, err = Dlatbs(Lower, Trans, NonUnit, normin, n, kd, ab, work, work.Off(2*n)); err != nil {
				panic(err)
			}
		}

		//        Multiply by 1/SCALE if doing so will not cause overflow.
		scale = scalel * scaleu
		if scale != one {
			ix = work.Iamax(n, 1)
			if scale < math.Abs(work.Get(ix-1))*smlnum || scale == zero {
				return
			}
			Drscl(n, scale, work, 1)
		}
		goto label10
	}

	//     Compute the estimate of the reciprocal condition number.
	if ainvnm != zero {
		rcond = (one / ainvnm) / anorm
	}

	return
}
