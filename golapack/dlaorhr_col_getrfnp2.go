package golapack

import (
	"fmt"
	"math"

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
// DlaorhrColGetrfnp2 is called to factorize a block by the blocked
// routine DLAORHR_COL_GETRFNP, which uses blocked code calling
//. Level 3 BLAS to update the submatrix. However, DlaorhrColGetrfnp2
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
func DlaorhrColGetrfnp2(m, n int, a *mat.Matrix, d *mat.Vector) (err error) {
	var one, sfmin float64
	var i, n1, n2 int

	one = 1.0

	//     Test the input parameters
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("DlaorhrColGetrfnp2", err)
		return
	}

	//     Quick return if possible
	if min(m, n) == 0 {
		return
	}
	if m == 1 {
		//        One row case, (also recursion termination case),
		//        use unblocked code
		//
		//        Transfer the sign
		d.Set(0, -math.Copysign(one, a.Get(0, 0)))

		//        Construct the row of U
		a.Set(0, 0, a.Get(0, 0)-d.Get(0))

	} else if n == 1 {
		//        One column case, (also recursion termination case),
		//        use unblocked code
		//
		//        Transfer the sign
		d.Set(0, -math.Copysign(one, a.Get(0, 0)))

		//        Construct the row of U
		a.Set(0, 0, a.Get(0, 0)-d.Get(0))

		//        Scale the elements 2:M of the column
		//
		//        Determine machine safe minimum
		sfmin = Dlamch(SafeMinimum)

		//        Construct the subdiagonal elements of L
		if math.Abs(a.Get(0, 0)) >= sfmin {
			a.Off(1, 0).Vector().Scal(m-1, one/a.Get(0, 0), 1)
		} else {
			for i = 2; i <= m; i++ {
				a.Set(i-1, 0, a.Get(i-1, 0)/a.Get(0, 0))
			}
		}

	} else {
		//        Divide the matrix B into four submatrices
		n1 = min(m, n) / 2
		n2 = n - n1

		//        Factor B11, recursive call
		if err = DlaorhrColGetrfnp2(n1, n1, a, d); err != nil {
			panic(err)
		}

		//        Solve for B21
		if err = a.Off(n1, 0).Trsm(Right, Upper, NoTrans, NonUnit, m-n1, n1, one, a); err != nil {
			panic(err)
		}

		//        Solve for B12
		if err = a.Off(0, n1).Trsm(Left, Lower, NoTrans, Unit, n1, n2, one, a); err != nil {
			panic(err)
		}

		//        Update B22, i.e. compute the Schur complement
		//        B22 := B22 - B21*B12
		if err = a.Off(n1, n1).Gemm(NoTrans, NoTrans, m-n1, n2, n1, -one, a.Off(n1, 0), a.Off(0, n1), one); err != nil {
			panic(err)
		}

		//        Factor B22, recursive call
		if err = DlaorhrColGetrfnp2(m-n1, n2, a.Off(n1, n1), d.Off(n1)); err != nil {
			panic(err)
		}

	}

	return
}
