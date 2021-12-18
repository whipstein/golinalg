package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpbtf2 computes the Cholesky factorization of a real symmetric
// positive definite band matrix A.
//
// The factorization has the form
//    A = U**T * U ,  if UPLO = 'U', or
//    A = L  * L**T,  if UPLO = 'L',
// where U is an upper triangular matrix, U**T is the transpose of U, and
// L is lower triangular.
//
// This is the unblocked version of the algorithm, calling Level 2 BLAS.
func Dpbtf2(uplo mat.MatUplo, n, kd int, ab *mat.Matrix) (info int, err error) {
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
		gltest.Xerbla2("Dpbtf2", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	kld = max(1, ab.Rows-1)

	if upper {
		//        Compute the Cholesky factorization A = U**T*U.
		for j = 1; j <= n; j++ {
			//           Compute U(J,J) and test for non-positive-definiteness.
			ajj = ab.Get(kd, j-1)
			if ajj <= zero {
				goto label30
			}
			ajj = math.Sqrt(ajj)
			ab.Set(kd, j-1, ajj)

			//           Compute elements J+1:J+KN of row J and update the
			//           trailing submatrix within the band.
			kn = min(kd, n-j)
			if kn > 0 {
				ab.Off(kd-1, j).Vector().Scal(kn, one/ajj, kld)
				err = ab.Off(kd, j).UpdateRows(kld).Syr(Upper, kn, -one, ab.Off(kd-1, j).Vector(), kld)
			}
		}
	} else {
		//        Compute the Cholesky factorization A = L*L**T.
		for j = 1; j <= n; j++ {
			//           Compute L(J,J) and test for non-positive-definiteness.
			ajj = ab.Get(0, j-1)
			if ajj <= zero {
				goto label30
			}
			ajj = math.Sqrt(ajj)
			ab.Set(0, j-1, ajj)

			//           Compute elements J+1:J+KN of column J and update the
			//           trailing submatrix within the band.
			kn = min(kd, n-j)
			if kn > 0 {
				ab.Off(1, j-1).Vector().Scal(kn, one/ajj, 1)
				err = ab.Off(0, j).UpdateRows(kld).Syr(Lower, kn, -one, ab.Off(1, j-1).Vector(), 1)
			}
		}
	}
	return

label30:
	;
	info = j

	return
}
