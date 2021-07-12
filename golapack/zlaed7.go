package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlaed7 computes the updated eigensystem of a diagonal
// matrix after modification by a rank-one symmetric matrix. This
// routine is used only for the eigenproblem which requires all
// eigenvalues and optionally eigenvectors of a dense or banded
// Hermitian matrix that has been reduced to tridiagonal form.
//
//   T = Q(in) ( D(in) + RHO * Z*Z**H ) Q**H(in) = Q(out) * D(out) * Q**H(out)
//
//   where Z = Q**Hu, u is a vector of length N with ones in the
//   CUTPNT and CUTPNT + 1 th elements and zeros elsewhere.
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
//       equation via the routine DLAED4 (as called by SLAED3).
//       This routine also calculates the eigenvectors of the current
//       problem.
//
//       The final stage consists of computing the updated eigenvectors
//       directly using the updated eigenvalues.  The eigenvectors for
//       the current problem are multiplied with the eigenvectors from
//       the overall problem.
func Zlaed7(n, cutpnt, qsiz, tlvls, curlvl, curpbm *int, d *mat.Vector, q *mat.CMatrix, ldq *int, rho *float64, indxq *[]int, qstore *mat.Vector, qptr, prmptr, perm, givptr, givcol *[]int, givnum *mat.Matrix, work *mat.CVector, rwork *mat.Vector, iwork *[]int, info *int) {
	var coltyp, curr, i, idlmda, indx, indxc, indxp, iq, iw, iz, k, n1, n2, ptr int

	//     Test the input parameters.
	(*info) = 0

	//     IF( ICOMPQ.LT.0 .OR. ICOMPQ.GT.1 ) THEN
	//        INFO = -1
	//     ELSE IF( N.LT.0 ) THEN
	if (*n) < 0 {
		(*info) = -1
	} else if min(int(1), *n) > (*cutpnt) || (*n) < (*cutpnt) {
		(*info) = -2
	} else if (*qsiz) < (*n) {
		(*info) = -3
	} else if (*ldq) < max(1, *n) {
		(*info) = -9
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZLAED7"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     The following values are for bookkeeping purposes only.  They are
	//     integer pointers which indicate the portion of the workspace
	//     used by a particular array in DLAED2 and SLAED3.
	iz = 1
	idlmda = iz + (*n)
	iw = idlmda + (*n)
	iq = iw + (*n)

	indx = 1
	indxc = indx + (*n)
	coltyp = indxc + (*n)
	indxp = coltyp + (*n)

	//     Form the z-vector which consists of the last row of Q_1 and the
	//     first row of Q_2.
	ptr = 1 + pow(2, *tlvls)
	for i = 1; i <= (*curlvl)-1; i++ {
		ptr = ptr + pow(2, (*tlvls)-i)
	}
	curr = ptr + (*curpbm)
	Dlaeda(n, tlvls, curlvl, curpbm, prmptr, perm, givptr, givcol, givnum, qstore, qptr, rwork.Off(iz-1), rwork.Off(iz+(*n)-1), info)

	//     When solving the final problem, we no longer need the stored data,
	//     so we will overwrite the data from this level onto the previously
	//     used storage space.
	if (*curlvl) == (*tlvls) {
		(*qptr)[curr-1] = 1
		(*prmptr)[curr-1] = 1
		(*givptr)[curr-1] = 1
	}

	//     Sort and Deflate eigenvalues.
	Zlaed8(&k, n, qsiz, q, ldq, d, rho, cutpnt, rwork.Off(iz-1), rwork.Off(idlmda-1), work.CMatrix(*qsiz, opts), qsiz, rwork.Off(iw-1), toSlice(iwork, indxp-1), toSlice(iwork, indx-1), indxq, toSlice(perm, (*prmptr)[curr-1]-1), &(*givptr)[curr], toSlice(givcol, 0+((*givptr)[curr-1]-1)*2), givnum.Off(0, (*givptr)[curr-1]-1), info)
	(*prmptr)[curr] = (*prmptr)[curr-1] + (*n)
	(*givptr)[curr] = (*givptr)[curr] + (*givptr)[curr-1]

	//     Solve Secular Equation.
	if k != 0 {
		Dlaed9(&k, func() *int { y := 1; return &y }(), &k, n, d, rwork.MatrixOff(iq-1, k, opts), &k, rho, rwork.Off(idlmda-1), rwork.Off(iw-1), qstore.MatrixOff((*qptr)[curr-1]-1, k, opts), &k, info)
		Zlacrm(qsiz, &k, work.CMatrix(*qsiz, opts), qsiz, qstore.MatrixOff((*qptr)[curr-1]-1, k, opts), &k, q, ldq, rwork.Off(iq-1))
		(*qptr)[curr] = (*qptr)[curr-1] + pow(k, 2)
		if (*info) != 0 {
			return
		}

		//     Prepare the INDXQ sorting premutation.
		n1 = k
		n2 = (*n) - k
		Dlamrg(&n1, &n2, d, func() *int { y := 1; return &y }(), toPtr(-1), indxq)
	} else {
		(*qptr)[curr] = (*qptr)[curr-1]
		for i = 1; i <= (*n); i++ {
			(*indxq)[i-1] = i
		}
	}
}
