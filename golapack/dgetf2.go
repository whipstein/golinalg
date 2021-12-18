package golapack

import (
	"fmt"
	"math"

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
func Dgetf2(m, n int, a *mat.Matrix, ipiv *[]int) (info int, err error) {
	var one, sfmin, zero float64
	var i, j, jp int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("Dgetf2", err)
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 {
		return
	}

	//     Compute machine safe minimum
	sfmin = Dlamch(SafeMinimum)
	//
	for j = 1; j <= min(m, n); j++ {
		//        Find pivot and test for singularity.
		jp = j - 1 + a.Off(j-1, j-1).Vector().Iamax(m-j+1, 1)
		(*ipiv)[j-1] = jp
		if a.Get(jp-1, j-1) != zero {
			//           Apply the interchange to columns 1:N.
			if jp != j {
				a.Off(jp-1, 0).Vector().Swap(n, a.Off(j-1, 0).Vector(), a.Rows, a.Rows)
			}

			//           Compute elements J+1:M of J-th column.
			if j < m {
				if math.Abs(a.Get(j-1, j-1)) >= sfmin {
					a.Off(j, j-1).Vector().Scal(m-j, one/a.Get(j-1, j-1), 1)
				} else {
					for i = 1; i <= m-j; i++ {
						a.Set(j+i-1, j-1, a.Get(j+i-1, j-1)/a.Get(j-1, j-1))
					}
				}
			}

		} else if info == 0 {
			info = j
		}

		if j < min(m, n) {
			//           Update trailing submatrix.
			err = a.Off(j, j).Ger(m-j, n-j, -one, a.Off(j, j-1).Vector(), 1, a.Off(j-1, j).Vector(), a.Rows)
		}
	}

	return
}
