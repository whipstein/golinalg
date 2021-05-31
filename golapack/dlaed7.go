package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlaed7 computes the updated eigensystem of a diagonal
// matrix after modification by a rank-one symmetric matrix. This
// routine is used only for the eigenproblem which requires all
// eigenvalues and optionally eigenvectors of a dense symmetric matrix
// that has been reduced to tridiagonal form.  DLAED1 handles
// the case in which all eigenvalues and eigenvectors of a symmetric
// tridiagonal matrix are desired.
//
//   T = Q(in) ( D(in) + RHO * Z*Z**T ) Q**T(in) = Q(out) * D(out) * Q**T(out)
//
//    where Z = Q**Tu, u is a vector of length N with ones in the
//    CUTPNT and CUTPNT + 1 th elements and zeros elsewhere.
//
//    The eigenvectors of the original matrix are stored in Q, and the
//    eigenvalues are in D.  The algorithm consists of three stages:
//
//       The first stage consists of deflating the size of the problem
//       when there are multiple eigenvalues or if there is a zero in
//       the Z vector.  For each such occurrence the dimension of the
//       secular equation problem is reduced by one.  This stage is
//       performed by the routine DLAED8.
//
//       The second stage consists of calculating the updated
//       eigenvalues. This is done by finding the roots of the secular
//       equation via the routine DLAED4 (as called by DLAED9).
//       This routine also calculates the eigenvectors of the current
//       problem.
//
//       The final stage consists of computing the updated eigenvectors
//       directly using the updated eigenvalues.  The eigenvectors for
//       the current problem are multiplied with the eigenvectors from
//       the overall problem.
func Dlaed7(icompq, n, qsiz, tlvls, curlvl, curpbm *int, d *mat.Vector, q *mat.Matrix, ldq *int, indxq *[]int, rho *float64, cutpnt *int, qstore *mat.Vector, qptr, prmptr, perm, givptr, givcol *[]int, givnum *mat.Matrix, work *mat.Vector, iwork *[]int, info *int) {
	var one, zero float64
	var coltyp, curr, i, idlmda, indx, indxc, indxp, iq2, is, iw, iz, k, ldq2, n1, n2, ptr int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	(*info) = 0

	if (*icompq) < 0 || (*icompq) > 1 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*icompq) == 1 && (*qsiz) < (*n) {
		(*info) = -3
	} else if (*ldq) < maxint(1, *n) {
		(*info) = -9
	} else if minint(1, *n) > (*cutpnt) || (*n) < (*cutpnt) {
		(*info) = -12
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLAED7"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     The following values are for bookkeeping purposes only.  They are
	//     integer pointers which indicate the portion of the workspace
	//     used by a particular array in DLAED8 and DLAED9.
	if (*icompq) == 1 {
		ldq2 = (*qsiz)
	} else {
		ldq2 = (*n)
	}

	iz = 1
	idlmda = iz + (*n)
	iw = idlmda + (*n)
	iq2 = iw + (*n)
	is = iq2 + (*n)*ldq2

	indx = 1
	indxc = indx + (*n)
	coltyp = indxc + (*n)
	indxp = coltyp + (*n)

	//     Form the z-vector which consists of the last row of Q_1 and the
	//     first row of Q_2.
	ptr = 1 + int(math.Pow(2, float64(*tlvls)))
	for i = 1; i <= (*curlvl)-1; i++ {
		ptr = ptr + int(math.Pow(2, float64((*tlvls)-i)))
	}
	curr = ptr + (*curpbm)
	Dlaeda(n, tlvls, curlvl, curpbm, prmptr, perm, givptr, givcol, givnum, qstore, qptr, work.Off(iz-1), work.Off(iz+(*n)-1), info)

	//     When solving the final problem, we no longer need the stored data,
	//     so we will overwrite the data from this level onto the previously
	//     used storage space.
	if (*curlvl) == (*tlvls) {
		(*qptr)[curr-1] = 1
		(*prmptr)[curr-1] = 1
		(*givptr)[curr-1] = 1
	}

	//     Sort and Deflate eigenvalues.
	Dlaed8(icompq, &k, n, qsiz, d, q, ldq, indxq, rho, cutpnt, work.Off(iz-1), work.Off(idlmda-1), work.MatrixOff(iq2-1, ldq2, opts), &ldq2, work.Off(iw-1), toSlice(perm, (*prmptr)[curr-1]-1), &((*givptr)[curr+1-1]), toSlice(givcol, 0+((*givptr)[curr-1]-1)*2), givnum.Off(0, (*givptr)[curr-1]-1), toSlice(iwork, indxp-1), toSlice(iwork, indx-1), info)
	(*prmptr)[curr+1-1] = (*prmptr)[curr-1] + (*n)
	(*givptr)[curr+1-1] = (*givptr)[curr+1-1] + (*givptr)[curr-1]

	//     Solve Secular Equation.
	if k != 0 {
		Dlaed9(&k, func() *int { y := 1; return &y }(), &k, n, d, work.MatrixOff(is-1, k, opts), &k, rho, work.Off(idlmda-1), work.Off(iw-1), qstore.MatrixOff((*qptr)[curr-1]-1, k, opts), &k, info)
		if (*info) != 0 {
			return
		}
		if (*icompq) == 1 {
			goblas.Dgemm(NoTrans, NoTrans, qsiz, &k, &k, &one, work.MatrixOff(iq2-1, ldq2, opts), &ldq2, qstore.MatrixOff((*qptr)[curr-1]-1, k, opts), &k, &zero, q, ldq)
		}
		(*qptr)[curr+1-1] = (*qptr)[curr-1] + int(math.Pow(float64(k), 2))

		//     Prepare the INDXQ sorting permutation.
		n1 = k
		n2 = (*n) - k
		Dlamrg(&n1, &n2, d, func() *int { y := 1; return &y }(), toPtr(-1), indxq)
	} else {
		(*qptr)[curr+1-1] = (*qptr)[curr-1]
		for i = 1; i <= (*n); i++ {
			(*indxq)[i-1] = i
		}
	}
}
