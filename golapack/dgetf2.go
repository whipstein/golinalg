package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgetf2 computes an LU factorization of a general m-by-n matrix A
// using partial pivoting with row interchanges.
//
// The factorization has the form
//    A = P * L * U
// where P is a permutation matrix, L is lower triangular with unit
// diagonal elements (lower trapezoidal if m > n), and U is upper
// triangular (upper trapezoidal if m < n).
//
// This is the right-looking Level 2 BLAS version of the algorithm.
func Dgetf2(m, n *int, a *mat.Matrix, lda *int, ipiv *[]int, info *int) {
	var one, sfmin, zero float64
	var i, j, jp int
	var err error
	_ = err

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *m) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGETF2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		return
	}

	//     Compute machine safe minimum
	sfmin = Dlamch(SafeMinimum)
	//
	for j = 1; j <= min(*m, *n); j++ {
		//        Find pivot and test for singularity.
		jp = j - 1 + goblas.Idamax((*m)-j+1, a.Vector(j-1, j-1, 1))
		(*ipiv)[j-1] = jp
		if a.Get(jp-1, j-1) != zero {
			//           Apply the interchange to columns 1:N.
			if jp != j {
				goblas.Dswap(*n, a.Vector(j-1, 0), a.Vector(jp-1, 0))
			}

			//           Compute elements J+1:M of J-th column.
			if j < (*m) {
				if math.Abs(a.Get(j-1, j-1)) >= sfmin {
					goblas.Dscal((*m)-j, one/a.Get(j-1, j-1), a.Vector(j, j-1, 1))
				} else {
					for i = 1; i <= (*m)-j; i++ {
						a.Set(j+i-1, j-1, a.Get(j+i-1, j-1)/a.Get(j-1, j-1))
					}
				}
			}

		} else if (*info) == 0 {
			*info = j
		}

		if j < min(*m, *n) {
			//           Update trailing submatrix.
			err = goblas.Dger((*m)-j, (*n)-j, -one, a.Vector(j, j-1, 1), a.Vector(j-1, j), a.Off(j, j))
		}
	}
}
