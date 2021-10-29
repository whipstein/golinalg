package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgetrf2 computes an LU factorization of a general M-by-N matrix A
// using partial pivoting with row interchanges.
//
// The factorization has the form
//    A = P * L * U
// where P is a permutation matrix, L is lower triangular with unit
// diagonal elements (lower trapezoidal if m > n), and U is upper
// triangular (upper trapezoidal if m < n).
//
// This is the recursive version of the algorithm. It divides
// the matrix into four submatrices:
//
//        [  A11 | A12  ]  where A11 is n1 by n1 and A22 is n2 by n2
//    A = [ -----|----- ]  with n1 = min(m,n)/2
//        [  A21 | A22  ]       n2 = n-n1
//
//                                       [ A11 ]
// The subroutine calls itself to factor [ --- ],
//                                       [ A12 ]
//                 [ A12 ]
// do the swaps on [ --- ], solve A12, update A22,
//                 [ A22 ]
//
// then calls itself to factor A22 and do the swaps on A21.
func Dgetrf2(m, n int, a *mat.Matrix, ipiv *[]int) (info int, err error) {
	var one, sfmin, temp, zero float64
	var i, iinfo, n1, n2 int

	one = 1.0
	zero = 0.0

	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("DGETRF2", err)
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 {
		return
	}
	if m == 1 {
		//        Use unblocked code for one row case
		//        Just need to handle IPIV and INFO
		(*ipiv)[0] = 1
		if a.Get(0, 0) == zero {
			info = 1
		}

	} else if n == 1 {
		//        Use unblocked code for one column case
		//
		//
		//        Compute machine safe minimum
		sfmin = Dlamch(SafeMinimum)

		//        Find pivot and test for singularity
		i = goblas.Idamax(m, a.Vector(0, 0, 1))
		(*ipiv)[0] = i
		if a.Get(i-1, 0) != zero {
			//           Apply the interchange
			if i != 1 {
				temp = a.Get(0, 0)
				a.Set(0, 0, a.Get(i-1, 0))
				a.Set(i-1, 0, temp)
			}

			//           Compute elements 2:M of the column
			if math.Abs(a.Get(0, 0)) >= sfmin {
				goblas.Dscal(m-1, one/a.Get(0, 0), a.Vector(1, 0, 1))
			} else {
				for i = 1; i <= m-1; i++ {
					a.Set(1+i-1, 0, a.Get(1+i-1, 0)/a.Get(0, 0))
				}
			}

		} else {
			info = 1
		}

	} else {
		//        Use recursive code
		n1 = min(m, n) / 2
		n2 = n - n1

		//               [ A11 ]
		//        Factor [ --- ]
		//               [ A21 ]
		if iinfo, err = Dgetrf2(m, n1, a, ipiv); info == 0 && iinfo > 0 {
			info = iinfo
		}

		//                              [ A12 ]
		//        Apply interchanges to [ --- ]
		//                              [ A22 ]
		Dlaswp(n2, a.Off(0, n1), 1, n1, (*ipiv), 1)

		//        Solve A12
		err = goblas.Dtrsm(mat.Left, mat.Lower, mat.NoTrans, mat.Unit, n1, n2, one, a, a.Off(0, n1))

		//        Update A22
		err = goblas.Dgemm(mat.NoTrans, mat.NoTrans, m-n1, n2, n1, -one, a.Off(n1, 0), a.Off(0, n1), one, a.Off(n1, n1))

		//        Factor A22
		iinfo, err = Dgetrf2(m-n1, n2, a.Off(n1, n1), toSlice(ipiv, n1))

		//        Adjust INFO and the pivot indices
		if info == 0 && iinfo > 0 {
			info = iinfo + n1
		}
		for i = n1 + 1; i <= min(m, n); i++ {
			(*ipiv)[i-1] += n1
		}

		//        Apply interchanges to A21
		Dlaswp(n1, a, n1+1, min(m, n), *ipiv, 1)

	}

	return
}
