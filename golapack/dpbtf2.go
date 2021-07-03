package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
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
func Dpbtf2(uplo byte, n, kd *int, ab *mat.Matrix, ldab, info *int) {
	var upper bool
	var ajj, one, zero float64
	var j, kld, kn int
	var err error
	_ = err

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*kd) < 0 {
		(*info) = -3
	} else if (*ldab) < (*kd)+1 {
		(*info) = -5
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DPBTF2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	kld = maxint(1, (*ldab)-1)

	if upper {
		//        Compute the Cholesky factorization A = U**T*U.
		for j = 1; j <= (*n); j++ {
			//           Compute U(J,J) and test for non-positive-definiteness.
			ajj = ab.Get((*kd)+1-1, j-1)
			if ajj <= zero {
				goto label30
			}
			ajj = math.Sqrt(ajj)
			ab.Set((*kd)+1-1, j-1, ajj)

			//           Compute elements J+1:J+KN of row J and update the
			//           trailing submatrix within the band.
			kn = minint(*kd, (*n)-j)
			if kn > 0 {
				goblas.Dscal(kn, one/ajj, ab.Vector((*kd)-1, j+1-1), kld)
				err = goblas.Dsyr(Upper, kn, -one, ab.Vector((*kd)-1, j+1-1), kld, ab.Off((*kd)+1-1, j+1-1).UpdateRows(kld), kld)
				ab.UpdateRows(*ldab)
			}
		}
	} else {
		//        Compute the Cholesky factorization A = L*L**T.
		for j = 1; j <= (*n); j++ {
			//           Compute L(J,J) and test for non-positive-definiteness.
			ajj = ab.Get(0, j-1)
			if ajj <= zero {
				goto label30
			}
			ajj = math.Sqrt(ajj)
			ab.Set(0, j-1, ajj)

			//           Compute elements J+1:J+KN of column J and update the
			//           trailing submatrix within the band.
			kn = minint(*kd, (*n)-j)
			if kn > 0 {
				goblas.Dscal(kn, one/ajj, ab.Vector(1, j-1), 1)
				err = goblas.Dsyr(Lower, kn, -one, ab.Vector(1, j-1), 1, ab.Off(0, j+1-1).UpdateRows(kld), kld)
				ab.UpdateRows(*ldab)
			}
		}
	}
	return

label30:
	;
	(*info) = j
}
