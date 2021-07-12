package golapack

import (
	"github.com/whipstein/golinalg/goblas"
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
func Dlaed1(n *int, d *mat.Vector, q *mat.Matrix, ldq *int, indxq *[]int, rho *float64, cutpnt *int, work *mat.Vector, iwork *[]int, info *int) {
	var coltyp, i, idlmda, indx, indxc, indxp, iq2, is, iw, iz, k, n1, n2, zpp1 int

	//     Test the input parameters.
	(*info) = 0

	if (*n) < 0 {
		(*info) = -1
	} else if (*ldq) < max(1, *n) {
		(*info) = -4
	} else if min(1, (*n)/2) > (*cutpnt) || ((*n)/2) < (*cutpnt) {
		(*info) = -7
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLAED1"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     The following values are integer pointers which indicate
	//     the portion of the workspace
	//     used by a particular array in DLAED2 and DLAED3.
	iz = 1
	idlmda = iz + (*n)
	iw = idlmda + (*n)
	iq2 = iw + (*n)

	indx = 1
	indxc = indx + (*n)
	coltyp = indxc + (*n)
	indxp = coltyp + (*n)

	//     Form the z-vector which consists of the last row of Q_1 and the
	//     first row of Q_2.
	goblas.Dcopy(*cutpnt, q.Vector((*cutpnt)-1, 0), work.Off(iz-1, 1))
	zpp1 = (*cutpnt) + 1
	goblas.Dcopy((*n)-(*cutpnt), q.Vector(zpp1-1, zpp1-1), work.Off(iz+(*cutpnt)-1, 1))

	//     Deflate eigenvalues.
	Dlaed2(&k, n, cutpnt, d, q, ldq, indxq, rho, work.Off(iz-1), work.Off(idlmda-1), work.Off(iw-1), work.Off(iq2-1), toSlice(iwork, indx-1), toSlice(iwork, indxc-1), toSlice(iwork, indxp-1), toSlice(iwork, coltyp-1), info)

	if (*info) != 0 {
		return
	}

	//     Solve Secular Equation.
	if k != 0 {
		is = ((*iwork)[coltyp-1]+(*iwork)[coltyp])*(*cutpnt) + ((*iwork)[coltyp]+(*iwork)[coltyp+2-1])*((*n)-(*cutpnt)) + iq2
		Dlaed3(&k, n, cutpnt, d, q, ldq, rho, work.Off(idlmda-1), work.Off(iq2-1), toSlice(iwork, indxc-1), toSlice(iwork, indxc-1), work.Off(iw-1), work.Off(is-1), info)
		if (*info) != 0 {
			return
		}

		//     Prepare the INDXQ sorting permutation.
		n1 = k
		n2 = (*n) - k
		Dlamrg(&n1, &n2, d, func() *int { y := 1; return &y }(), toPtr(-1), indxq)
	} else {
		for i = 1; i <= (*n); i++ {
			(*indxq)[i-1] = i
		}
	}
}
