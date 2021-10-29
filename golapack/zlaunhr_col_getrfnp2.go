package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// ZlaunhrColGetrfnp2 computes the modified LU factorization without
// pivoting of a complex general M-by-N matrix A. The factorization has
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
// reconstruction routine ZUNHR_COL. In ZUNHR_COL, this routine is
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
// ZlaunhrColGetrfnp2 is called to factorize a block by the blocked
// routine ZLAUNHR_COL_GETRFNP, which uses blocked code calling
//. Level 3 BLAS to update the submatrix. However, ZlaunhrColGetrfnp2
// is self-sufficient and can be used without ZLAUNHR_COL_GETRFNP.
//
// [1] "Reconstructing Householder vectors from tall-skinny QR",
//     G. Ballard, J. Demmel, L. Grigori, M. Jacquelin, H.D. Nguyen,
//     E. Solomonik, J. Parallel Distrib. Comput.,
//     vol. 85, pp. 3-31, 2015.
//
// [2] "Recursion leads to automatic variable blocking for dense linear
//     algebra algorithms", F. Gustavson, IBM J. of Res. and Dev.,
//     vol. 41, no. 6, pp. 737-755, 1997.
func ZlaunhrColGetrfnp2(m, n int, a *mat.CMatrix, d *mat.CVector) (err error) {
	var cone complex128
	var one, sfmin float64
	var i, n1, n2 int

	one = 1.0
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("ZlaunhrColGetrfnp2", err)
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
		d.SetRe(0, -math.Copysign(one, a.GetRe(0, 0)))

		//        Construct the row of U
		a.Set(0, 0, a.Get(0, 0)-d.Get(0))

	} else if n == 1 {
		//        One column case, (also recursion termination case),
		//        use unblocked code
		//
		//        Transfer the sign
		d.SetRe(0, -math.Copysign(one, a.GetRe(0, 0)))

		//        Construct the row of U
		a.Set(0, 0, a.Get(0, 0)-d.Get(0))

		//        Scale the elements 2:M of the column
		//
		//        Determine machine safe minimum
		sfmin = Dlamch(SafeMinimum)

		//        Construct the subdiagonal elements of L
		if cabs1(a.Get(0, 0)) >= sfmin {
			goblas.Zscal(m-1, cone/a.Get(0, 0), a.CVector(1, 0, 1))
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
		if err = ZlaunhrColGetrfnp2(n1, n1, a, d); err != nil {
			panic(err)
		}

		//        Solve for B21
		if err = goblas.Ztrsm(Right, Upper, NoTrans, NonUnit, m-n1, n1, cone, a, a.Off(n1, 0)); err != nil {
			panic(err)
		}

		//        Solve for B12
		if err = goblas.Ztrsm(Left, Lower, NoTrans, Unit, n1, n2, cone, a, a.Off(0, n1)); err != nil {
			panic(err)
		}

		//        Update B22, i.e. compute the Schur complement
		//        B22 := B22 - B21*B12
		if err = goblas.Zgemm(NoTrans, NoTrans, m-n1, n2, n1, -cone, a.Off(n1, 0), a.Off(0, n1), cone, a.Off(n1, n1)); err != nil {
			panic(err)
		}

		//        Factor B22, recursive call
		if err = ZlaunhrColGetrfnp2(m-n1, n2, a.Off(n1, n1), d.Off(n1)); err != nil {
			panic(err)
		}

	}

	return
}
