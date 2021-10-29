package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgtcon estimates the reciprocal of the condition number of a complex
// tridiagonal matrix A using the LU factorization as computed by
// ZGTTRF.
//
// An estimate is obtained for norm(inv(A)), and the reciprocal of the
// condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
func Zgtcon(norm byte, n int, dl, d, du, du2 *mat.CVector, ipiv *[]int, anorm float64, work *mat.CVector) (rcond float64, err error) {
	var onenrm bool
	var ainvnm, one, zero float64
	var i, kase, kase1 int

	isave := make([]int, 3)

	one = 1.0
	zero = 0.0

	//     Test the input arguments.
	onenrm = norm == '1' || norm == 'O'
	if !onenrm && norm != 'I' {
		err = fmt.Errorf("!onenrm && norm != 'I': norm='%c'", norm)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if anorm < zero {
		err = fmt.Errorf("anorm < zero: anorm=%v", anorm)
	}
	if err != nil {
		gltest.Xerbla2("Zgtcon", err)
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

	//     Check that D(1:N) is non-zero.
	for i = 1; i <= n; i++ {
		if d.Get(i-1) == complex(zero, 0) {
			return
		}
	}

	ainvnm = zero
	if onenrm {
		kase1 = 1
	} else {
		kase1 = 2
	}
	kase = 0
label20:
	;
	ainvnm, kase = Zlacn2(n, work.Off(n), work, ainvnm, kase, &isave)
	if kase != 0 {
		if kase == kase1 {
			//           Multiply by inv(U)*inv(L).
			if err = Zgttrs(NoTrans, n, 1, dl, d, du, du2, ipiv, work.CMatrix(n, opts)); err != nil {
				panic(err)
			}
		} else {
			//           Multiply by inv(L**H)*inv(U**H).
			if err = Zgttrs(ConjTrans, n, 1, dl, d, du, du2, ipiv, work.CMatrix(n, opts)); err != nil {
				panic(err)
			}
		}
		goto label20
	}

	//     Compute the estimate of the reciprocal condition number.
	if ainvnm != zero {
		rcond = (one / ainvnm) / anorm
	}

	return
}
