package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsyconrook estimates the reciprocal of the condition number (in the
// 1-norm) of a complex symmetric matrix A using the factorization
// A = U*D*U**T or A = L*D*L**T computed by ZSYTRF_ROOK.
//
// An estimate is obtained for norm(inv(A)), and the reciprocal of the
// condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
func ZsyconRook(uplo mat.MatUplo, n int, a *mat.CMatrix, ipiv *[]int, anorm float64, work *mat.CVector) (rcond float64, err error) {
	var upper bool
	var czero complex128
	var ainvnm, one, zero float64
	var i, kase int

	isave := make([]int, 3)

	one = 1.0
	zero = 0.0
	czero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if anorm < zero {
		err = fmt.Errorf("anorm < zero: anorm=%v", anorm)
	}
	if err != nil {
		gltest.Xerbla2("ZsyconRook", err)
		return
	}

	//     Quick return if possible
	rcond = zero
	if n == 0 {
		rcond = one
		return
	} else if anorm <= zero {
		return
	}

	//     Check that the diagonal matrix D is nonsingular.
	if upper {
		//        Upper triangular storage: examine D from bottom to top
		for i = n; i >= 1; i-- {
			if (*ipiv)[i-1] > 0 && a.Get(i-1, i-1) == czero {
				return
			}
		}
	} else {
		//        Lower triangular storage: examine D from top to bottom.
		for i = 1; i <= n; i++ {
			if (*ipiv)[i-1] > 0 && a.Get(i-1, i-1) == czero {
				return
			}
		}
	}

	//     Estimate the 1-norm of the inverse.
	kase = 0
label30:
	;
	ainvnm, kase = Zlacn2(n, work.Off(n), work, ainvnm, kase, &isave)
	if kase != 0 {
		//        Multiply by inv(L*D*L**T) or inv(U*D*U**T).
		if err = ZsytrsRook(uplo, n, 1, a, ipiv, work.CMatrix(n, opts)); err != nil {
			panic(err)
		}
		goto label30
	}

	//     Compute the estimate of the reciprocal condition number.
	if ainvnm != zero {
		rcond = (one / ainvnm) / anorm
	}

	return
}
