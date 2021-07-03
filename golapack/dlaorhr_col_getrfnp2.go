package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// DlaorhrColGetrfnp2 computes the modified LU factorization without
// pivoting of a real general M-by-N matrix A. The factorization has
// the form:
//
//     A - S = L * U,
//
// where:
//    S is a m-by-n diagonal sign matrix with the diagonal D, so that
//    D(i) = S(i,i), 1 <= i <= min(M,N). The diagonal D is constructed
//    as D(i)=-SIGN(A(i,i)), where A(i,i) is the value after performing
//    i-1 steps of Gaussian elimination. This means that the diagonal
//    element at each step of "modified" Gaussian elimination is at
//    least one in absolute value (so that division-by-zero not
//    possible during the division by the diagonal element);
//
//    L is a M-by-N lower triangular matrix with unit diagonal elements
//    (lower trapezoidal if M > N);
//
//    and U is a M-by-N upper triangular matrix
//    (upper trapezoidal if M < N).
//
// This routine is an auxiliary routine used in the Householder
// reconstruction routine DORHR_COL. In DORHR_COL, this routine is
// applied to an M-by-N matrix A with orthonormal columns, where each
// element is bounded by one in absolute value. With the choice of
// the matrix S above, one can show that the diagonal element at each
// step of Gaussian elimination is the largest (in absolute value) in
// the column on or below the diagonal, so that no pivoting is required
// for numerical stability [1].
//
// For more details on the Householder reconstruction algorithm,
// including the modified LU factorization, see [1].
//
// This is the recursive version of the LU factorization algorithm.
// Denote A - S by B. The algorithm divides the matrix B into four
// submatrices:
//
//        [  B11 | B12  ]  where B11 is n1 by n1,
//    B = [ -----|----- ]        B21 is (m-n1) by n1,
//        [  B21 | B22  ]        B12 is n1 by n2,
//                               B22 is (m-n1) by n2,
//                               with n1 = min(m,n)/2, n2 = n-n1.
//
//
// The subroutine calls itself to factor B11, solves for B21,
// solves for B12, updates B22, then calls itself to factor B22.
//
// For more details on the recursive LU algorithm, see [2].
//
// DLAORHR_COL_GETRFNP2 is called to factorize a block by the blocked
// routine DLAORHR_COL_GETRFNP, which uses blocked code calling
//. Level 3 BLAS to update the submatrix. However, DLAORHR_COL_GETRFNP2
// is self-sufficient and can be used without DLAORHR_COL_GETRFNP.
//
// [1] "Reconstructing Householder vectors from tall-skinny QR",
//     G. Ballard, J. Demmel, L. Grigori, M. Jacquelin, H.D. Nguyen,
//     E. Solomonik, J. Parallel Distrib. Comput.,
//     vol. 85, pp. 3-31, 2015.
//
// [2] "Recursion leads to automatic variable blocking for dense linear
//     algebra algorithms", F. Gustavson, IBM J. of Res. and Dev.,
//     vol. 41, no. 6, pp. 737-755, 1997.
func DlaorhrColGetrfnp2(m, n *int, a *mat.Matrix, lda *int, d *mat.Vector, info *int) {
	var one, sfmin float64
	var i, iinfo, n1, n2 int
	var err error
	_ = err

	one = 1.0

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
		gltest.Xerbla([]byte("DLAORHR_COL_GETRFNP2"), -(*info))
		return
	}

	//     Quick return if possible
	if minint(*m, *n) == 0 {
		return
	}
	if (*m) == 1 {
		//        One row case, (also recursion termination case),
		//        use unblocked code
		//
		//        Transfer the sign
		d.Set(0, -signf64(one, a.Get(0, 0)))

		//        Construct the row of U
		a.Set(0, 0, a.Get(0, 0)-d.Get(0))

	} else if (*n) == 1 {
		//        One column case, (also recursion termination case),
		//        use unblocked code
		//
		//        Transfer the sign
		d.Set(0, -signf64(one, a.Get(0, 0)))

		//        Construct the row of U
		a.Set(0, 0, a.Get(0, 0)-d.Get(0))

		//        Scale the elements 2:M of the column
		//
		//        Determine machine safe minimum
		sfmin = Dlamch(SafeMinimum)

		//        Construct the subdiagonal elements of L
		if math.Abs(a.Get(0, 0)) >= sfmin {
			goblas.Dscal((*m)-1, one/a.Get(0, 0), a.Vector(1, 0), 1)
		} else {
			for i = 2; i <= (*m); i++ {
				a.Set(i-1, 0, a.Get(i-1, 0)/a.Get(0, 0))
			}
		}

	} else {
		//        Divide the matrix B into four submatrices
		n1 = minint(*m, *n) / 2
		n2 = (*n) - n1

		//        Factor B11, recursive call
		DlaorhrColGetrfnp2(&n1, &n1, a, lda, d, &iinfo)

		//        Solve for B21
		err = goblas.Dtrsm(Right, Upper, NoTrans, NonUnit, (*m)-n1, n1, one, a, *lda, a.Off(n1+1-1, 0), *lda)

		//        Solve for B12
		err = goblas.Dtrsm(Left, Lower, NoTrans, Unit, n1, n2, one, a, *lda, a.Off(0, n1+1-1), *lda)

		//        Update B22, i.e. compute the Schur complement
		//        B22 := B22 - B21*B12
		err = goblas.Dgemm(NoTrans, NoTrans, (*m)-n1, n2, n1, -one, a.Off(n1+1-1, 0), *lda, a.Off(0, n1+1-1), *lda, one, a.Off(n1+1-1, n1+1-1), *lda)

		//        Factor B22, recursive call
		DlaorhrColGetrfnp2(toPtr((*m)-n1), &n2, a.Off(n1+1-1, n1+1-1), lda, d.Off(n1+1-1), &iinfo)

	}
}
