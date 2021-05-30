package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dpptrf computes the Cholesky factorization of a real symmetric
// positive definite matrix A stored in packed format.
//
// The factorization has the form
//    A = U**T * U,  if UPLO = 'U', or
//    A = L  * L**T,  if UPLO = 'L',
// where U is an upper triangular matrix and L is lower triangular.
func Dpptrf(uplo byte, n *int, ap *mat.Vector, info *int) {
	var upper bool
	var ajj, one, zero float64
	var j, jc, jj int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DPPTRF"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if upper {
		//        Compute the Cholesky factorization A = U**T*U.
		jj = 0
		for j = 1; j <= (*n); j++ {
			jc = jj + 1
			jj = jj + j

			//           Compute elements 1:J-1 of column J.
			if j > 1 {
				goblas.Dtpsv(mat.Upper, mat.Trans, mat.NonUnit, toPtr(j-1), ap, ap.Off(jc-1), toPtr(1))
			}

			//           Compute U(J,J) and test for non-positive-definiteness.
			ajj = ap.Get(jj-1) - goblas.Ddot(toPtr(j-1), ap.Off(jc-1), toPtr(1), ap.Off(jc-1), toPtr(1))
			if ajj <= zero {
				ap.Set(jj-1, ajj)
				goto label30
			}
			ap.Set(jj-1, math.Sqrt(ajj))
		}
	} else {
		//        Compute the Cholesky factorization A = L*L**T.
		jj = 1
		for j = 1; j <= (*n); j++ {
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
			if j < (*n) {
				goblas.Dscal(toPtr((*n)-j), toPtrf64(one/ajj), ap.Off(jj+1-1), toPtr(1))
				goblas.Dspr(mat.Lower, toPtr((*n)-j), toPtrf64(-one), ap.Off(jj+1-1), toPtr(1), ap.Off(jj+(*n)-j+1-1))
				jj = jj + (*n) - j + 1
			}
		}
	}
	return

label30:
	;
	(*info) = j
}
