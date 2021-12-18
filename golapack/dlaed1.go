package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlaed1 computes the updated eigensystem of a diagonal
// matrix after modification by a rank-one symmetric matrix.  This
// routine is used only for the eigenproblem which requires all
// eigenvalues and eigenvectors of a tridiagonal matrix.  DLAED7 handles
// the case in which eigenvalues only or eigenvalues and eigenvectors
// of a full symmetric matrix (which was reduced to tridiagonal form)
// are desired.
//
//   T = Q(in) ( D(in) + RHO * Z*Z**T ) Q**T(in) = Q(out) * D(out) * Q**T(out)
//
//    where Z = Q**T*u, u is a vector of length N with ones in the
//    CUTPNT and CUTPNT + 1 th elements and zeros elsewhere.
//
//    The eigenvectors of the original matrix are stored in Q, and the
//    eigenvalues are in D.  The algorithm consists of three stages:
//
//       The first stage consists of deflating the size of the problem
//       when there are multiple eigenvalues or if there is a zero in
//       the Z vector.  For each such occurrence the dimension of the
//       secular equation problem is reduced by one.  This stage is
//       performed by the routine DLAED2.
//
//       The second stage consists of calculating the updated
//       eigenvalues. This is done by finding the roots of the secular
//       equation via the routine DLAED4 (as called by DLAED3).
//       This routine also calculates the eigenvectors of the current
//       problem.
//
//       The final stage consists of computing the updated eigenvectors
//       directly using the updated eigenvalues.  The eigenvectors for
//       the current problem are multiplied with the eigenvectors from
//       the overall problem.
func Dlaed1(n int, d *mat.Vector, q *mat.Matrix, indxq *[]int, rho float64, cutpnt int, work *mat.Vector, iwork *[]int) (info int, err error) {
	var coltyp, i, idlmda, indx, indxc, indxp, iq2, is, iw, iz, k, n1, n2, zpp1 int

	//     Test the input parameters.
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if q.Rows < max(1, n) {
		err = fmt.Errorf("q.Rows < max(1, n): q.Rows=%v, n=%v", q.Rows, n)
	} else if min(1, n/2) > cutpnt || (n/2) < cutpnt {
		err = fmt.Errorf("min(1, n/2) > cutpnt || (n/2) < cutpnt: n=%v, cutpnt=%v", n, cutpnt)
	}
	if err != nil {
		gltest.Xerbla2("Dlaed1", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     The following values are integer pointers which indicate
	//     the portion of the workspace
	//     used by a particular array in DLAED2 and DLAED3.
	iz = 1
	idlmda = iz + n
	iw = idlmda + n
	iq2 = iw + n

	indx = 1
	indxc = indx + n
	coltyp = indxc + n
	indxp = coltyp + n

	//     Form the z-vector which consists of the last row of Q_1 and the
	//     first row of Q_2.
	work.Off(iz-1).Copy(cutpnt, q.Off(cutpnt-1, 0).Vector(), q.Rows, 1)
	zpp1 = cutpnt + 1
	work.Off(iz+cutpnt-1).Copy(n-cutpnt, q.Off(zpp1-1, zpp1-1).Vector(), q.Rows, 1)

	//     Deflate eigenvalues.
	if rho, err = Dlaed2(k, n, cutpnt, d, q, indxq, rho, work.Off(iz-1), work.Off(idlmda-1), work.Off(iw-1), work.Off(iq2-1), toSlice(iwork, indx-1), toSlice(iwork, indxc-1), toSlice(iwork, indxp-1), toSlice(iwork, coltyp-1)); err != nil {
		panic(err)
	}

	if info != 0 {
		return
	}

	//     Solve Secular Equation.
	if k != 0 {
		is = ((*iwork)[coltyp-1]+(*iwork)[coltyp])*cutpnt + ((*iwork)[coltyp]+(*iwork)[coltyp+2-1])*(n-cutpnt) + iq2
		if info, err = Dlaed3(k, n, cutpnt, d, q, rho, work.Off(idlmda-1), work.Off(iq2-1), toSlice(iwork, indxc-1), toSlice(iwork, indxc-1), work.Off(iw-1), work.Off(is-1)); err != nil {
			panic(err)
		}
		if info != 0 {
			return
		}

		//     Prepare the INDXQ sorting permutation.
		n1 = k
		n2 = n - k
		Dlamrg(n1, n2, d, 1, -1, indxq)
	} else {
		for i = 1; i <= n; i++ {
			(*indxq)[i-1] = i
		}
	}

	return
}
