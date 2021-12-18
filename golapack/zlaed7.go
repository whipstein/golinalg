package golapack

import (
	"fmt"

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
func Zlaed7(n, cutpnt, qsiz, tlvls, curlvl, curpbm int, d *mat.Vector, q *mat.CMatrix, rho float64, indxq *[]int, qstore *mat.Vector, qptr, prmptr, perm, givptr, givcol *[]int, givnum *mat.Matrix, work *mat.CVector, rwork *mat.Vector, iwork *[]int) (info int, err error) {
	var coltyp, curr, i, idlmda, indx, indxc, indxp, iq, iw, iz, k, n1, n2, ptr int

	//     Test the input parameters.
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if min(1, n) > cutpnt || n < cutpnt {
		err = fmt.Errorf("min(1, n) > cutpnt || n < cutpnt: n=%v, cutpnt=%v", n, cutpnt)
	} else if qsiz < n {
		err = fmt.Errorf("qsiz < n: qsiz=%v, n=%v", qsiz, n)
	} else if q.Rows < max(1, n) {
		err = fmt.Errorf("q.Rows < max(1, n): q.Rows=%v, n=%v", q.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zlaed7", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     The following values are for bookkeeping purposes only.  They are
	//     integer pointers which indicate the portion of the workspace
	//     used by a particular array in DLAED2 and SLAED3.
	iz = 1
	idlmda = iz + n
	iw = idlmda + n
	iq = iw + n

	indx = 1
	indxc = indx + n
	coltyp = indxc + n
	indxp = coltyp + n

	//     Form the z-vector which consists of the last row of Q_1 and the
	//     first row of Q_2.
	ptr = 1 + pow(2, tlvls)
	for i = 1; i <= curlvl-1; i++ {
		ptr = ptr + pow(2, tlvls-i)
	}
	curr = ptr + curpbm
	if err = Dlaeda(n, tlvls, curlvl, curpbm, prmptr, perm, givptr, givcol, givnum, qstore, qptr, rwork.Off(iz-1), rwork.Off(iz+n-1)); err != nil {
		panic(err)
	}

	//     When solving the final problem, we no longer need the stored data,
	//     so we will overwrite the data from this level onto the previously
	//     used storage space.
	if curlvl == tlvls {
		(*qptr)[curr-1] = 1
		(*prmptr)[curr-1] = 1
		(*givptr)[curr-1] = 1
	}

	//     Sort and Deflate eigenvalues.
	if k, rho, (*givptr)[curr], err = Zlaed8(n, qsiz, q, d, rho, cutpnt, rwork.Off(iz-1), rwork.Off(idlmda-1), work.CMatrix(qsiz, opts), rwork.Off(iw-1), toSlice(iwork, indxp-1), toSlice(iwork, indx-1), indxq, toSlice(perm, (*prmptr)[curr-1]-1), toSlice(givcol, 0+((*givptr)[curr-1]-1)*2), givnum.Off(0, (*givptr)[curr-1]-1)); err != nil {
		panic(err)
	}
	(*prmptr)[curr] = (*prmptr)[curr-1] + n
	(*givptr)[curr] = (*givptr)[curr] + (*givptr)[curr-1]

	//     Solve Secular Equation.
	if k != 0 {
		if info, err = Dlaed9(k, 1, k, n, d, rwork.Off(iq-1).Matrix(k, opts), rho, rwork.Off(idlmda-1), rwork.Off(iw-1), qstore.Off((*qptr)[curr-1]-1).Matrix(k, opts)); err != nil {
			panic(err)
		}
		Zlacrm(qsiz, k, work.CMatrix(qsiz, opts), qstore.Off((*qptr)[curr-1]-1).Matrix(k, opts), q, rwork.Off(iq-1))
		(*qptr)[curr] = (*qptr)[curr-1] + pow(k, 2)
		if info != 0 {
			return
		}

		//     Prepare the INDXQ sorting premutation.
		n1 = k
		n2 = n - k
		Dlamrg(n1, n2, d, 1, -1, indxq)
	} else {
		(*qptr)[curr] = (*qptr)[curr-1]
		for i = 1; i <= n; i++ {
			(*indxq)[i-1] = i
		}
	}

	return
}
