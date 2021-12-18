package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpptrf computes the Cholesky factorization of a real symmetric
// positive definite matrix A stored in packed format.
//
// The factorization has the form
//    A = U**T * U,  if UPLO = 'U', or
//    A = L  * L**T,  if UPLO = 'L',
// where U is an upper triangular matrix and L is lower triangular.
func Dpptrf(uplo mat.MatUplo, n int, ap *mat.Vector) (info int, err error) {
	var upper bool
	var ajj, one, zero float64
	var j, jc, jj int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	}
	if err != nil {
		gltest.Xerbla2("Dpptrf", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if upper {
		//        Compute the Cholesky factorization A = U**T*U.
		jj = 0
		for j = 1; j <= n; j++ {
			jc = jj + 1
			jj = jj + j

			//           Compute elements 1:J-1 of column J.
			if j > 1 {
				err = ap.Off(jc-1).Tpsv(Upper, Trans, NonUnit, j-1, ap, 1)
			}

			//           Compute U(J,J) and test for non-positive-definiteness.
			ajj = ap.Get(jj-1) - ap.Off(jc-1).Dot(j-1, ap.Off(jc-1), 1, 1)
			if ajj <= zero {
				ap.Set(jj-1, ajj)
				goto label30
			}
			ap.Set(jj-1, math.Sqrt(ajj))
		}
	} else {
		//        Compute the Cholesky factorization A = L*L**T.
		jj = 1
		for j = 1; j <= n; j++ {
			//           Compute L(J,J) and test for non-positive-definiteness.
			ajj = ap.Get(jj - 1)
			if ajj <= zero {
				ap.Set(jj-1, ajj)
				goto label30
			}
			ajj = math.Sqrt(ajj)
			ap.Set(jj-1, ajj)

			//           Compute elements J+1:N of column J and update the trailing
			//           submatrix.
			if j < n {
				ap.Off(jj).Scal(n-j, one/ajj, 1)
				err = ap.Off(jj+n-j).Spr(mat.Lower, n-j, -one, ap.Off(jj), 1)
				jj = jj + n - j + 1
			}
		}
	}
	return

label30:
	;
	info = j

	return
}
