package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
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
func Zgetf2(m, n int, a *mat.CMatrix, ipiv *[]int) (info int, err error) {
	var one, zero complex128
	var sfmin float64
	var i, j, jp int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("Zgetf2", err)
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 {
		return
	}

	//     Compute machine safe minimum
	sfmin = Dlamch(SafeMinimum)

	for j = 1; j <= min(m, n); j++ {
		//        Find pivot and test for singularity.
		jp = j - 1 + a.Off(j-1, j-1).CVector().Iamax(m-j+1, 1)
		(*ipiv)[j-1] = jp
		if a.Get(jp-1, j-1) != zero {
			//           Apply the interchange to columns 1:N.
			if jp != j {
				a.Off(jp-1, 0).CVector().Swap(n, a.Off(j-1, 0).CVector(), a.Rows, a.Rows)
			}

			//           Compute elements J+1:M of J-th column.
			if j < m {
				if a.GetMag(j-1, j-1) >= sfmin {
					a.Off(j, j-1).CVector().Scal(m-j, one/a.Get(j-1, j-1), 1)
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
			err = a.Off(j, j).Geru(m-j, n-j, -one, a.Off(j, j-1).CVector(), 1, a.Off(j-1, j).CVector(), a.Rows)
		}
	}

	return
}
