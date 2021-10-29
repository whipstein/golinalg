package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsycon3 estimates the reciprocal of the condition number (in the
// 1-norm) of a real symmetric matrix A using the factorization
// computed by DSYTRF_RK or DSYTRF_BK:
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
// This routine uses BLAS3 solver DSYTRS_3.
func Dsycon3(uplo mat.MatUplo, n int, a *mat.Matrix, e *mat.Vector, ipiv *[]int, anorm float64, work *mat.Vector, iwork *[]int) (rcond float64, err error) {
	var upper bool
	var ainvnm, one, zero float64
	var i, kase int

	isave := make([]int, 3)

	one = 1.0
	zero = 0.0

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
		gltest.Xerbla2("Dsycon3", err)
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
			if (*ipiv)[i-1] > 0 && a.Get(i-1, i-1) == zero {
				return
			}
		}
	} else {
		//        Lower triangular storage: examine D from top to bottom.
		for i = 1; i <= n; i++ {
			if (*ipiv)[i-1] > 0 && a.Get(i-1, i-1) == zero {
				return
			}
		}
	}

	//     Estimate the 1-norm of the inverse.
	kase = 0
label30:
	;
	ainvnm, kase = Dlacn2(n, work.Off(n), work, iwork, ainvnm, kase, &isave)
	if kase != 0 {
		//        Multiply by inv(L*D*L**T) or inv(U*D*U**T).
		if _, err = Dsytrs3(uplo, n, 1, a, e, ipiv, work.Matrix(n, opts)); err != nil {
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
