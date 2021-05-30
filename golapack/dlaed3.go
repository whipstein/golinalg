package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dlaed3 finds the roots of the secular equation, as defined by the
// values in D, W, and RHO, between 1 and K.  It makes the
// appropriate calls to DLAED4 and then updates the eigenvectors by
// multiplying the matrix of eigenvectors of the pair of eigensystems
// being combined by the matrix of eigenvectors of the K-by-K system
// which is solved here.
//
// This code makes very mild assumptions about floating point
// arithmetic. It will work on machines with a guard digit in
// add/subtract, or on those binary machines without guard digits
// which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
// It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Dlaed3(k, n, n1 *int, d *mat.Vector, q *mat.Matrix, ldq *int, rho *float64, dlamda, q2 *mat.Vector, indx, ctot *[]int, w, s *mat.Vector, info *int) {
	var one, temp, zero float64
	var i, ii, iq2, j, n12, n2, n23 int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	(*info) = 0

	if (*k) < 0 {
		(*info) = -1
	} else if (*n) < (*k) {
		(*info) = -2
	} else if (*ldq) < maxint(1, *n) {
		(*info) = -6
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLAED3"), -(*info))
		return
	}

	//     Quick return if possible
	if (*k) == 0 {
		return
	}

	//     Modify values DLAMDA(i) to make sure all DLAMDA(i)-DLAMDA(j) can
	//     be computed with high relative accuracy (barring over/underflow).
	//     This is a problem on machines without a guard digit in
	//     add/subtract (Cray XMP, Cray YMP, Cray C 90 and Cray 2).
	//     The following code replaces DLAMDA(I) by 2*DLAMDA(I)-DLAMDA(I),
	//     which on any of these machines zeros out the bottommost
	//     bit of DLAMDA(I) if it is 1; this makes the subsequent
	//     subtractions DLAMDA(I)-DLAMDA(J) unproblematic when cancellation
	//     occurs. On binary machines with a guard digit (almost all
	//     machines) it does not change DLAMDA(I) at all. On hexadecimal
	//     and decimal machines with a guard digit, it slightly
	//     changes the bottommost bits of DLAMDA(I). It does not account
	//     for hexadecimal or decimal machines without guard digits
	//     (we know of none). We use a subroutine call to compute
	//     2*DLAMBDA(I) to prevent optimizing compilers from eliminating
	//     this code.
	for i = 1; i <= (*k); i++ {
		dlamda.Set(i-1, Dlamc3(dlamda.GetPtr(i-1), dlamda.GetPtr(i-1))-dlamda.Get(i-1))
	}

	for j = 1; j <= (*k); j++ {
		Dlaed4(k, &j, dlamda, w, q.Vector(0, j-1), rho, d.GetPtr(j-1), info)

		//        If the zero finder fails, the computation is terminated.
		if (*info) != 0 {
			return
		}
	}

	if (*k) == 1 {
		goto label110
	}
	if (*k) == 2 {
		for j = 1; j <= (*k); j++ {
			w.Set(0, q.Get(0, j-1))
			w.Set(1, q.Get(1, j-1))
			ii = (*indx)[0]
			q.Set(0, j-1, w.Get(ii-1))
			ii = (*indx)[1]
			q.Set(1, j-1, w.Get(ii-1))
		}
		goto label110
	}

	//     Compute updated W.
	goblas.Dcopy(k, w, toPtr(1), s, toPtr(1))

	//     Initialize W(I) = Q(I,I)
	goblas.Dcopy(k, q.VectorIdx(0), toPtr((*ldq)+1), w, toPtr(1))
	for j = 1; j <= (*k); j++ {
		for i = 1; i <= j-1; i++ {
			w.Set(i-1, w.Get(i-1)*(q.Get(i-1, j-1)/(dlamda.Get(i-1)-dlamda.Get(j-1))))
		}
		for i = j + 1; i <= (*k); i++ {
			w.Set(i-1, w.Get(i-1)*(q.Get(i-1, j-1)/(dlamda.Get(i-1)-dlamda.Get(j-1))))
		}
	}
	for i = 1; i <= (*k); i++ {
		w.Set(i-1, signf64(math.Sqrt(-w.Get(i-1)), s.Get(i-1)))
	}

	//     Compute eigenvectors of the modified rank-1 modification.
	for j = 1; j <= (*k); j++ {
		for i = 1; i <= (*k); i++ {
			s.Set(i-1, w.Get(i-1)/q.Get(i-1, j-1))
		}
		temp = goblas.Dnrm2(k, s, toPtr(1))
		for i = 1; i <= (*k); i++ {
			ii = (*indx)[i-1]
			q.Set(i-1, j-1, s.Get(ii-1)/temp)
		}
	}

	//     Compute the updated eigenvectors.
label110:
	;

	n2 = (*n) - (*n1)
	n12 = (*ctot)[0] + (*ctot)[1]
	n23 = (*ctot)[1] + (*ctot)[2]

	Dlacpy('A', &n23, k, q.Off((*ctot)[0]+1-1, 0), ldq, s.Matrix(n23, opts), &n23)
	iq2 = (*n1)*n12 + 1
	if n23 != 0 {
		goblas.Dgemm(NoTrans, NoTrans, &n2, k, &n23, &one, q2.MatrixOff(iq2-1, n2, opts), &n2, s.Matrix(n23, opts), &n23, &zero, q.Off((*n1)+1-1, 0), ldq)
	} else {
		Dlaset('A', &n2, k, &zero, &zero, q.Off((*n1)+1-1, 0), ldq)
	}

	Dlacpy('A', &n12, k, q, ldq, s.Matrix(n12, opts), &n12)
	if n12 != 0 {
		goblas.Dgemm(NoTrans, NoTrans, n1, k, &n12, &one, q2.Matrix(*n1, opts), n1, s.Matrix(n12, opts), &n12, &zero, q, ldq)
	} else {
		Dlaset('A', n1, k, &zero, &zero, q.Off(0, 0), ldq)
	}
}
