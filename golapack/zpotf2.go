package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zpotf2 computes the Cholesky factorization of a complex Hermitian
// positive definite matrix A.
//
// The factorization has the form
//    A = U**H * U ,  if UPLO = 'U', or
//    A = L  * L**H,  if UPLO = 'L',
// where U is an upper triangular matrix and L is lower triangular.
//
// This is the unblocked version of the algorithm, calling Level 2 BLAS.
func Zpotf2(uplo mat.MatUplo, n int, a *mat.CMatrix) (info int, err error) {
	var upper bool
	var cone complex128
	var ajj, one, zero float64
	var j int

	one = 1.0
	zero = 0.0
	cone = (1.0 + 0.0*1i)

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
		gltest.Xerbla2("Zpotf2", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if upper {
		//        Compute the Cholesky factorization A = U**H *U.
		for j = 1; j <= n; j++ {
			//           Compute U(J,J) and test for non-positive-definiteness.
			ajj = a.GetRe(j-1, j-1) - real(a.Off(0, j-1).CVector().Dotc(j-1, a.Off(0, j-1).CVector(), 1, 1))
			if ajj <= zero || Disnan(int(ajj)) {
				a.SetRe(j-1, j-1, ajj)
				goto label30
			}
			ajj = math.Sqrt(ajj)
			a.SetRe(j-1, j-1, ajj)

			//           Compute elements J+1:N of row J.
			if j < n {
				Zlacgv(j-1, a.Off(0, j-1).CVector(), 1)
				if err = a.Off(j-1, j).CVector().Gemv(Trans, j-1, n-j, -cone, a.Off(0, j), a.Off(0, j-1).CVector(), 1, cone, a.Rows); err != nil {
					panic(err)
				}
				Zlacgv(j-1, a.Off(0, j-1).CVector(), 1)
				a.Off(j-1, j).CVector().Dscal(n-j, one/ajj, a.Rows)
			}
		}
	} else {
		//        Compute the Cholesky factorization A = L*L**H.
		for j = 1; j <= n; j++ {
			//           Compute L(J,J) and test for non-positive-definiteness.
			ajj = a.GetRe(j-1, j-1) - real(a.Off(j-1, 0).CVector().Dotc(j-1, a.Off(j-1, 0).CVector(), a.Rows, a.Rows))
			if ajj <= zero || Disnan(int(ajj)) {
				a.SetRe(j-1, j-1, ajj)
				goto label30
			}
			ajj = math.Sqrt(ajj)
			a.SetRe(j-1, j-1, ajj)

			//           Compute elements J+1:N of column J.
			if j < n {
				Zlacgv(j-1, a.Off(j-1, 0).CVector(), a.Rows)
				if err = a.Off(j, j-1).CVector().Gemv(NoTrans, n-j, j-1, -cone, a.Off(j, 0), a.Off(j-1, 0).CVector(), a.Rows, cone, 1); err != nil {
					panic(err)
				}
				Zlacgv(j-1, a.Off(j-1, 0).CVector(), a.Rows)
				a.Off(j, j-1).CVector().Dscal(n-j, one/ajj, 1)
			}
		}
	}
	return

label30:
	;
	info = j

	return
}
