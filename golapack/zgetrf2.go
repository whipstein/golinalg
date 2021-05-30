package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zgetrf2 computes an LU factorization of a general M-by-N matrix A
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
func Zgetrf2(m, n *int, a *mat.CMatrix, lda *int, ipiv *[]int, info *int) {
	var one, temp, zero complex128
	var sfmin float64
	var i, iinfo, n1, n2 int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *m) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGETRF2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		return
	}
	if (*m) == 1 {
		//        Use unblocked code for one row case
		//        Just need to handle IPIV and INFO
		(*ipiv)[0] = 1
		if a.Get(0, 0) == zero {
			(*info) = 1
		}

	} else if (*n) == 1 {
		//        Use unblocked code for one column case
		//
		//
		//        Compute machine safe minimum
		sfmin = Dlamch(SafeMinimum)

		//        Find pivot and test for singularity
		i = goblas.Izamax(m, a.CVector(0, 0), func() *int { y := 1; return &y }())
		(*ipiv)[0] = i
		if a.Get(i-1, 0) != zero {
			//           Apply the interchange
			if i != 1 {
				temp = a.Get(0, 0)
				a.Set(0, 0, a.Get(i-1, 0))
				a.Set(i-1, 0, temp)
			}

			//           Compute elements 2:M of the column
			if a.GetMag(0, 0) >= sfmin {
				goblas.Zscal(toPtr((*m)-1), toPtrc128(one/a.Get(0, 0)), a.CVector(1, 0), func() *int { y := 1; return &y }())
			} else {
				for i = 1; i <= (*m)-1; i++ {
					a.Set(1+i-1, 0, a.Get(1+i-1, 0)/a.Get(0, 0))
				}
			}

		} else {
			(*info) = 1
		}
	} else {
		//        Use recursive code
		n1 = minint(*m, *n) / 2
		n2 = (*n) - n1

		//               [ A11 ]
		//        Factor [ --- ]
		//               [ A21 ]
		Zgetrf2(m, &n1, a, lda, ipiv, &iinfo)
		if (*info) == 0 && iinfo > 0 {
			(*info) = iinfo
		}

		//                              [ A12 ]
		//        Apply interchanges to [ --- ]
		//                              [ A22 ]
		Zlaswp(&n2, a.Off(0, n1+1-1), lda, func() *int { y := 1; return &y }(), &n1, ipiv, func() *int { y := 1; return &y }())

		//        Solve A12
		goblas.Ztrsm(Left, Lower, NoTrans, Unit, &n1, &n2, &one, a, lda, a.Off(0, n1+1-1), lda)

		//        Update A22
		goblas.Zgemm(NoTrans, NoTrans, toPtr((*m)-n1), &n2, &n1, toPtrc128(-one), a.Off(n1+1-1, 0), lda, a.Off(0, n1+1-1), lda, &one, a.Off(n1+1-1, n1+1-1), lda)

		//        Factor A22
		Zgetrf2(toPtr((*m)-n1), &n2, a.Off(n1+1-1, n1+1-1), lda, toSlice(ipiv, n1+1-1), &iinfo)

		//        Adjust INFO and the pivot indices
		if (*info) == 0 && iinfo > 0 {
			(*info) = iinfo + n1
		}
		for i = n1 + 1; i <= minint(*m, *n); i++ {
			(*ipiv)[i-1] = (*ipiv)[i-1] + n1
		}

		//        Apply interchanges to A21
		Zlaswp(&n1, a.Off(0, 0), lda, toPtr(n1+1), toPtr(minint(*m, *n)), ipiv, func() *int { y := 1; return &y }())

	}
}