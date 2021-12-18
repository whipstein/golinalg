package golapack

import (
	"fmt"
	"math"

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
func Dlaed7(icompq, n, qsiz, tlvls, curlvl, curpbm int, d *mat.Vector, q *mat.Matrix, indxq *[]int, rho float64, cutpnt int, qstore *mat.Vector, qptr, prmptr, perm, givptr, givcol *[]int, givnum *mat.Matrix, work *mat.Vector, iwork *[]int) (info int, err error) {
	var one, zero float64
	var coltyp, curr, i, idlmda, indx, indxc, indxp, iq2, is, iw, iz, k, ldq2, n1, n2, ptr int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	if icompq < 0 || icompq > 1 {
		err = fmt.Errorf("icompq < 0 || icompq > 1: icompq=%v", icompq)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if icompq == 1 && qsiz < n {
		err = fmt.Errorf("icompq == 1 && qsiz < n: icompq=%v, n=%v, qsiz=%v", icompq, n, qsiz)
	} else if q.Rows < max(1, n) {
		err = fmt.Errorf("q.Rows < max(1, n): q.Rows=%v, n=%v", q.Rows, n)
	} else if min(1, n) > cutpnt || n < cutpnt {
		err = fmt.Errorf("min(1, n) > cutpnt || n < cutpnt: n=%v, cutpnt=%v", n, cutpnt)
	}
	if err != nil {
		gltest.Xerbla2("Dlaed7", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     The following values are for bookkeeping purposes only.  They are
	//     integer pointers which indicate the portion of the workspace
	//     used by a particular array in DLAED8 and DLAED9.
	if icompq == 1 {
		ldq2 = qsiz
	} else {
		ldq2 = n
	}

	iz = 1
	idlmda = iz + n
	iw = idlmda + n
	iq2 = iw + n
	is = iq2 + n*ldq2

	indx = 1
	indxc = indx + n
	coltyp = indxc + n
	indxp = coltyp + n

	//     Form the z-vector which consists of the last row of Q_1 and the
	//     first row of Q_2.
	ptr = 1 + int(math.Pow(2, float64(tlvls)))
	for i = 1; i <= curlvl-1; i++ {
		ptr = ptr + int(math.Pow(2, float64(tlvls-i)))
	}
	curr = ptr + curpbm
	if err = Dlaeda(n, tlvls, curlvl, curpbm, prmptr, perm, givptr, givcol, givnum, qstore, qptr, work.Off(iz-1), work.Off(iz+n-1)); err != nil {
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
	if k, rho, (*givptr)[curr], err = Dlaed8(icompq, n, qsiz, d, q, indxq, rho, cutpnt, work.Off(iz-1), work.Off(idlmda-1), work.Off(iq2-1).Matrix(ldq2, opts), work.Off(iw-1), toSlice(perm, (*prmptr)[curr-1]-1), toSlice(givcol, 0+((*givptr)[curr-1]-1)*2), givnum.Off(0, (*givptr)[curr-1]-1), toSlice(iwork, indxp-1), toSlice(iwork, indx-1)); err != nil {
		panic(err)
	}
	(*prmptr)[curr] = (*prmptr)[curr-1] + n
	(*givptr)[curr] = (*givptr)[curr] + (*givptr)[curr-1]

	//     Solve Secular Equation.
	if k != 0 {
		if info, err = Dlaed9(k, 1, k, n, d, work.Off(is-1).Matrix(k, opts), rho, work.Off(idlmda-1), work.Off(iw-1), qstore.Off((*qptr)[curr-1]-1).Matrix(k, opts)); err != nil {
			panic(err)
		}
		if info != 0 {
			return
		}
		if icompq == 1 {
			if err = q.Gemm(NoTrans, NoTrans, qsiz, k, k, one, work.Off(iq2-1).Matrix(ldq2, opts), qstore.Off((*qptr)[curr-1]-1).Matrix(k, opts), zero); err != nil {
				panic(err)
			}
		}
		(*qptr)[curr] = (*qptr)[curr-1] + int(math.Pow(float64(k), 2))

		//     Prepare the INDXQ sorting permutation.
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
