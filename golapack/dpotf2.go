package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
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
func Dpotf2(uplo byte, n *int, a *mat.Matrix, lda, info *int) {
	var upper bool
	var ajj, one, zero float64
	var j int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *n) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DPOTF2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if upper {
		//        Compute the Cholesky factorization A = U**T *U.
		for j = 1; j <= (*n); j++ {
			//           Compute U(J,J) and test for non-positive-definiteness.
			ajj = a.Get(j-1, j-1) - goblas.Ddot(toPtr(j-1), a.Vector(0, j-1), toPtr(1), a.Vector(0, j-1), toPtr(1))
			if ajj <= zero || Disnan(int(ajj)) {
				a.Set(j-1, j-1, ajj)
				goto label30
			}
			ajj = math.Sqrt(ajj)
			a.Set(j-1, j-1, ajj)

			//           Compute elements J+1:N of row J.
			if j < (*n) {
				goblas.Dgemv(mat.Trans, toPtr(j-1), toPtr((*n)-j), toPtrf64(-one), a.Off(0, j+1-1), lda, a.Vector(0, j-1), toPtr(1), &one, a.Vector(j-1, j+1-1), lda)
				goblas.Dscal(toPtr((*n)-j), toPtrf64(one/ajj), a.Vector(j-1, j+1-1), lda)
			}
		}
	} else {
		//        Compute the Cholesky factorization A = L*L**T.
		for j = 1; j <= (*n); j++ {
			//           Compute L(J,J) and test for non-positive-definiteness.
			ajj = a.Get(j-1, j-1) - goblas.Ddot(toPtr(j-1), a.Vector(j-1, 0), lda, a.Vector(j-1, 0), lda)
			if ajj <= zero || Disnan(int(ajj)) {
				a.Set(j-1, j-1, ajj)
				goto label30
			}
			ajj = math.Sqrt(ajj)
			a.Set(j-1, j-1, ajj)

			//           Compute elements J+1:N of column J.
			if j < (*n) {
				goblas.Dgemv(mat.NoTrans, toPtr((*n)-j), toPtr(j-1), toPtrf64(-one), a.Off(j+1-1, 0), lda, a.Vector(j-1, 0), lda, &one, a.Vector(j+1-1, j-1), toPtr(1))
				goblas.Dscal(toPtr((*n)-j), toPtrf64(one/ajj), a.Vector(j+1-1, j-1), toPtr(1))
			}
		}
	}
	goto label40

label30:
	;
	(*info) = j

label40:
}
