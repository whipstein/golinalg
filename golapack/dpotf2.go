package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpotf2 computes the Cholesky factorization of a real symmetric
// positive definite matrix A.
//
// The factorization has the form
//    A = U**T * U ,  if UPLO = 'U', or
//    A = L  * L**T,  if UPLO = 'L',
// where U is an upper triangular matrix and L is lower triangular.
//
// This is the unblocked version of the algorithm, calling Level 2 BLAS.
func Dpotf2(uplo mat.MatUplo, n int, a *mat.Matrix) (info int, err error) {
	var upper bool
	var ajj, one, zero float64
	var j int

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
	}
	if err != nil {
		gltest.Xerbla2("Dpotf2", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if upper {
		//        Compute the Cholesky factorization A = U**T *U.
		for j = 1; j <= n; j++ {
			//           Compute U(J,J) and test for non-positive-definiteness.
			ajj = a.Get(j-1, j-1) - a.Off(0, j-1).Vector().Dot(j-1, a.Off(0, j-1).Vector(), 1, 1)
			if ajj <= zero || Disnan(int(ajj)) {
				a.Set(j-1, j-1, ajj)
				goto label30
			}
			ajj = math.Sqrt(ajj)
			a.Set(j-1, j-1, ajj)

			//           Compute elements J+1:N of row J.
			if j < n {
				err = a.Off(j-1, j).Vector().Gemv(Trans, j-1, n-j, -one, a.Off(0, j), a.Off(0, j-1).Vector(), 1, one, a.Rows)
				a.Off(j-1, j).Vector().Scal(n-j, one/ajj, a.Rows)
			}
		}
	} else {
		//        Compute the Cholesky factorization A = L*L**T.
		for j = 1; j <= n; j++ {
			//           Compute L(J,J) and test for non-positive-definiteness.
			ajj = a.Get(j-1, j-1) - a.Off(j-1, 0).Vector().Dot(j-1, a.Off(j-1, 0).Vector(), a.Rows, a.Rows)
			if ajj <= zero || Disnan(int(ajj)) {
				a.Set(j-1, j-1, ajj)
				goto label30
			}
			ajj = math.Sqrt(ajj)
			a.Set(j-1, j-1, ajj)

			//           Compute elements J+1:N of column J.
			if j < n {
				err = a.Off(j, j-1).Vector().Gemv(NoTrans, n-j, j-1, -one, a.Off(j, 0), a.Off(j-1, 0).Vector(), a.Rows, one, 1)
				a.Off(j, j-1).Vector().Scal(n-j, one/ajj, 1)
			}
		}
	}
	goto label40

label30:
	;
	info = j

label40:

	return
}
