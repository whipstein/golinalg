package golapack

import (
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
func Zpbtf2(uplo byte, n, kd *int, ab *mat.CMatrix, ldab, info *int) {
	var upper bool
	var ajj, one, zero float64
	var j, kld, kn int

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
		gltest.Xerbla([]byte("ZPBTF2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	kld = maxint(1, (*ldab)-1)

	if upper {
		//        Compute the Cholesky factorization A = U**H * U.
		for j = 1; j <= (*n); j++ {
			//           Compute U(J,J) and test for non-positive-definiteness.
			ajj = ab.GetRe((*kd)+1-1, j-1)
			if ajj <= zero {
				ab.SetRe((*kd)+1-1, j-1, ajj)
				goto label30
			}
			ajj = math.Sqrt(ajj)
			ab.SetRe((*kd)+1-1, j-1, ajj)

			//           Compute elements J+1:J+KN of row J and update the
			//           trailing submatrix within the band.
			kn = minint(*kd, (*n)-j)
			if kn > 0 {
				goblas.Zdscal(&kn, toPtrf64(one/ajj), ab.CVector((*kd)-1, j+1-1), &kld)
				Zlacgv(&kn, ab.CVector((*kd)-1, j+1-1), &kld)
				goblas.Zher(Upper, &kn, toPtrf64(-one), ab.CVector((*kd)-1, j+1-1), &kld, ab.Off((*kd)+1-1, j+1-1).UpdateRows(kld), &kld)
				Zlacgv(&kn, ab.CVector((*kd)-1, j+1-1), &kld)
			}
		}
	} else {
		//        Compute the Cholesky factorization A = L*L**H.
		for j = 1; j <= (*n); j++ {
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
			kn = minint(*kd, (*n)-j)
			if kn > 0 {
				goblas.Zdscal(&kn, toPtrf64(one/ajj), ab.CVector(1, j-1), func() *int { y := 1; return &y }())
				goblas.Zher(Lower, &kn, toPtrf64(-one), ab.CVector(1, j-1), func() *int { y := 1; return &y }(), ab.Off(0, j+1-1).UpdateRows(kld), &kld)
			}
		}
	}
	return

label30:
	;
	(*info) = j
}
