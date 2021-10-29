package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zpptrf computes the Cholesky factorization of a complex Hermitian
// positive definite matrix A stored in packed format.
//
// The factorization has the form
//    A = U**H * U,  if UPLO = 'U', or
//    A = L  * L**H,  if UPLO = 'L',
// where U is an upper triangular matrix and L is lower triangular.
func Zpptrf(uplo mat.MatUplo, n int, ap *mat.CVector) (info int, err error) {
	var upper bool
	var ajj, one, zero float64
	var j, jc, jj int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	}
	if err != nil {
		gltest.Xerbla2("Zpptrf", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if upper {
		//        Compute the Cholesky factorization A = U**H * U.
		jj = 0
		for j = 1; j <= n; j++ {
			jc = jj + 1
			jj = jj + j

			//           Compute elements 1:J-1 of column J.
			if j > 1 {
				if err = goblas.Ztpsv(Upper, ConjTrans, NonUnit, j-1, ap, ap.Off(jc-1, 1)); err != nil {
					panic(err)
				}
			}

			//           Compute U(J,J) and test for non-positive-definiteness.
			ajj = real(ap.Get(jj-1) - goblas.Zdotc(j-1, ap.Off(jc-1, 1), ap.Off(jc-1, 1)))
			if ajj <= zero {
				ap.SetRe(jj-1, ajj)
				goto label30
			}
			ap.SetRe(jj-1, math.Sqrt(ajj))
		}
	} else {
		//        Compute the Cholesky factorization A = L * L**H.
		jj = 1
		for j = 1; j <= n; j++ {
			//           Compute L(J,J) and test for non-positive-definiteness.
			ajj = ap.GetRe(jj - 1)
			if ajj <= zero {
				ap.SetRe(jj-1, ajj)
				goto label30
			}
			ajj = math.Sqrt(ajj)
			ap.SetRe(jj-1, ajj)

			//           Compute elements J+1:N of column J and update the trailing
			//           submatrix.
			if j < n {
				goblas.Zdscal(n-j, one/ajj, ap.Off(jj, 1))
				if err = goblas.Zhpr(Lower, n-j, -one, ap.Off(jj, 1), ap.Off(jj+n-j)); err != nil {
					panic(err)
				}
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
