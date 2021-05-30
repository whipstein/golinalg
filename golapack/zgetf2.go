package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zgetf2 computes an LU factorization of a general m-by-n matrix A
// using partial pivoting with row interchanges.
//
// The factorization has the form
//    A = P * L * U
// where P is a permutation matrix, L is lower triangular with unit
// diagonal elements (lower trapezoidal if m > n), and U is upper
// triangular (upper trapezoidal if m < n).
//
// This is the right-looking Level 2 BLAS version of the algorithm.
func Zgetf2(m, n *int, a *mat.CMatrix, lda *int, ipiv *[]int, info *int) {
	var one, zero complex128
	var sfmin float64
	var i, j, jp int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *m) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGETF2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		return
	}

	//     Compute machine safe minimum
	sfmin = Dlamch(SafeMinimum)

	for j = 1; j <= minint(*m, *n); j++ {
		//        Find pivot and test for singularity.
		jp = j - 1 + goblas.Izamax(toPtr((*m)-j+1), a.CVector(j-1, j-1), func() *int { y := 1; return &y }())
		(*ipiv)[j-1] = jp
		if a.Get(jp-1, j-1) != zero {
			//           Apply the interchange to columns 1:N.
			if jp != j {
				goblas.Zswap(n, a.CVector(j-1, 0), lda, a.CVector(jp-1, 0), lda)
			}

			//           Compute elements J+1:M of J-th column.
			if j < (*m) {
				if a.GetMag(j-1, j-1) >= sfmin {
					goblas.Zscal(toPtr((*m)-j), toPtrc128(one/a.Get(j-1, j-1)), a.CVector(j+1-1, j-1), func() *int { y := 1; return &y }())
				} else {
					for i = 1; i <= (*m)-j; i++ {
						a.Set(j+i-1, j-1, a.Get(j+i-1, j-1)/a.Get(j-1, j-1))
					}
				}
			}

		} else if (*info) == 0 {

			(*info) = j
		}

		if j < minint(*m, *n) {
			//           Update trailing submatrix.
			goblas.Zgeru(toPtr((*m)-j), toPtr((*n)-j), toPtrc128(-one), a.CVector(j+1-1, j-1), func() *int { y := 1; return &y }(), a.CVector(j-1, j+1-1), lda, a.Off(j+1-1, j+1-1), lda)
		}
	}
}
