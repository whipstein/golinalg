package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zpbtf2 computes the Cholesky factorization of a complex Hermitian
// positive definite band matrix A.
//
// The factorization has the form
//    A = U**H * U ,  if UPLO = 'U', or
//    A = L  * L**H,  if UPLO = 'L',
// where U is an upper triangular matrix, U**H is the conjugate transpose
// of U, and L is lower triangular.
//
// This is the unblocked version of the algorithm, calling Level 2 BLAS.
func Zpbtf2(uplo mat.MatUplo, n, kd int, ab *mat.CMatrix) (info int, err error) {
	var upper bool
	var ajj, one, zero float64
	var j, kld, kn int

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
	}
	if err != nil {
		gltest.Xerbla2("Zpbtf2", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	kld = max(1, ab.Rows-1)

	if upper {
		//        Compute the Cholesky factorization A = U**H * U.
		for j = 1; j <= n; j++ {
			//           Compute U(J,J) and test for non-positive-definiteness.
			ajj = ab.GetRe(kd, j-1)
			if ajj <= zero {
				ab.SetRe(kd, j-1, ajj)
				goto label30
			}
			ajj = math.Sqrt(ajj)
			ab.SetRe(kd, j-1, ajj)

			//           Compute elements J+1:J+KN of row J and update the
			//           trailing submatrix within the band.
			kn = min(kd, n-j)
			if kn > 0 {
				goblas.Zdscal(kn, one/ajj, ab.CVector(kd-1, j, kld))
				Zlacgv(kn, ab.CVector(kd-1, j, kld))
				if err = goblas.Zher(Upper, kn, -one, ab.CVector(kd-1, j, kld), ab.Off(kd, j).UpdateRows(kld)); err != nil {
					panic(err)
				}
				Zlacgv(kn, ab.CVector(kd-1, j, kld))
			}
		}
	} else {
		//        Compute the Cholesky factorization A = L*L**H.
		for j = 1; j <= n; j++ {
			//           Compute L(J,J) and test for non-positive-definiteness.
			ajj = ab.GetRe(0, j-1)
			if ajj <= zero {
				ab.SetRe(0, j-1, ajj)
				goto label30
			}
			ajj = math.Sqrt(ajj)
			ab.SetRe(0, j-1, ajj)

			//           Compute elements J+1:J+KN of column J and update the
			//           trailing submatrix within the band.
			kn = min(kd, n-j)
			if kn > 0 {
				goblas.Zdscal(kn, one/ajj, ab.CVector(1, j-1, 1))
				if err = goblas.Zher(Lower, kn, -one, ab.CVector(1, j-1, 1), ab.Off(0, j).UpdateRows(kld)); err != nil {
					panic(err)
				}
			}
		}
	}
	return

label30:
	;
	info = j

	return
}
