package golapack

import (
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
func Zpptrf(uplo byte, n *int, ap *mat.CVector, info *int) {
	var upper bool
	var ajj, one, zero float64
	var j, jc, jj int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZPPTRF"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if upper {
		//        Compute the Cholesky factorization A = U**H * U.
		jj = 0
		for j = 1; j <= (*n); j++ {
			jc = jj + 1
			jj = jj + j

			//           Compute elements 1:J-1 of column J.
			if j > 1 {
				goblas.Ztpsv(Upper, ConjTrans, NonUnit, toPtr(j-1), ap, ap.Off(jc-1), func() *int { y := 1; return &y }())
			}

			//           Compute U(J,J) and test for non-positive-definiteness.
			ajj = real(ap.Get(jj-1) - goblas.Zdotc(toPtr(j-1), ap.Off(jc-1), func() *int { y := 1; return &y }(), ap.Off(jc-1), func() *int { y := 1; return &y }()))
			if ajj <= zero {
				ap.SetRe(jj-1, ajj)
				goto label30
			}
			ap.SetRe(jj-1, math.Sqrt(ajj))
		}
	} else {
		//        Compute the Cholesky factorization A = L * L**H.
		jj = 1
		for j = 1; j <= (*n); j++ {
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
			if j < (*n) {
				goblas.Zdscal(toPtr((*n)-j), toPtrf64(one/ajj), ap.Off(jj+1-1), func() *int { y := 1; return &y }())
				goblas.Zhpr(Lower, toPtr((*n)-j), toPtrf64(-one), ap.Off(jj+1-1), func() *int { y := 1; return &y }(), ap.Off(jj+(*n)-j+1-1))
				jj = jj + (*n) - j + 1
			}
		}
	}
	return

label30:
	;
	(*info) = j
}
