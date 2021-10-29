package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsycon3 estimates the reciprocal of the condition number (in the
// 1-norm) of a complex symmetric matrix A using the factorization
// computed by ZSYTRF_RK or ZSYTRF_BK:
//
//    A = P*U*D*(U**T)*(P**T) or A = P*L*D*(L**T)*(P**T),
//
// where U (or L) is unit upper (or lower) triangular matrix,
// U**T (or L**T) is the transpose of U (or L), P is a permutation
// matrix, P**T is the transpose of P, and D is symmetric and block
// diagonal with 1-by-1 and 2-by-2 diagonal blocks.
//
// An estimate is obtained for norm(inv(A)), and the reciprocal of the
// condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
// This routine uses BLAS3 solver ZSYTRS_3.
func Zsycon3(uplo mat.MatUplo, n int, a *mat.CMatrix, e *mat.CVector, ipiv *[]int, anorm float64, work *mat.CVector) (rcond float64, err error) {
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
		gltest.Xerbla2("Zsycon3", err)
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
		if err = Zsytrs3(uplo, n, 1, a, e, ipiv, work.CMatrix(n, opts)); err != nil {
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
