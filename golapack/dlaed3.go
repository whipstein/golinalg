package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
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
func Dlaed3(k, n, n1 int, d *mat.Vector, q *mat.Matrix, rho float64, dlamda, q2 *mat.Vector, indx, ctot *[]int, w, s *mat.Vector) (info int, err error) {
	var one, temp, zero float64
	var i, ii, iq2, j, n12, n2, n23 int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	if k < 0 {
		err = fmt.Errorf("k < 0: k=%v", k)
	} else if n < k {
		err = fmt.Errorf("n < k: k=%v, n=%v", k, n)
	} else if q.Rows < max(1, n) {
		err = fmt.Errorf("q.Rows < max(1, n): q.Rows=%v, n=%v", q.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dlaed3", err)
		return
	}

	//     Quick return if possible
	if k == 0 {
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
	for i = 1; i <= k; i++ {
		dlamda.Set(i-1, Dlamc3(dlamda.GetPtr(i-1), dlamda.GetPtr(i-1))-dlamda.Get(i-1))
	}

	for j = 1; j <= k; j++ {
		_d := d.GetPtr(j - 1)
		*_d, info = Dlaed4(k, j, dlamda, w, q.Vector(0, j-1), rho)

		//        If the zero finder fails, the computation is terminated.
		if info != 0 {
			return
		}
	}

	if k == 1 {
		goto label110
	}
	if k == 2 {
		for j = 1; j <= k; j++ {
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
	goblas.Dcopy(k, w, s)

	//     Initialize W(I) = Q(I,I)
	goblas.Dcopy(k, q.VectorIdx(0, q.Rows+1), w)
	for j = 1; j <= k; j++ {
		for i = 1; i <= j-1; i++ {
			w.Set(i-1, w.Get(i-1)*(q.Get(i-1, j-1)/(dlamda.Get(i-1)-dlamda.Get(j-1))))
		}
		for i = j + 1; i <= k; i++ {
			w.Set(i-1, w.Get(i-1)*(q.Get(i-1, j-1)/(dlamda.Get(i-1)-dlamda.Get(j-1))))
		}
	}
	for i = 1; i <= k; i++ {
		w.Set(i-1, math.Copysign(math.Sqrt(-w.Get(i-1)), s.Get(i-1)))
	}

	//     Compute eigenvectors of the modified rank-1 modification.
	for j = 1; j <= k; j++ {
		for i = 1; i <= k; i++ {
			s.Set(i-1, w.Get(i-1)/q.Get(i-1, j-1))
		}
		temp = goblas.Dnrm2(k, s)
		for i = 1; i <= k; i++ {
			ii = (*indx)[i-1]
			q.Set(i-1, j-1, s.Get(ii-1)/temp)
		}
	}

	//     Compute the updated eigenvectors.
label110:
	;

	n2 = n - n1
	n12 = (*ctot)[0] + (*ctot)[1]
	n23 = (*ctot)[1] + (*ctot)[2]

	Dlacpy(Full, n23, k, q.Off((*ctot)[0], 0), s.Matrix(n23, opts))
	iq2 = n1*n12 + 1
	if n23 != 0 {
		if err = goblas.Dgemm(NoTrans, NoTrans, n2, k, n23, one, q2.MatrixOff(iq2-1, n2, opts), s.Matrix(n23, opts), zero, q.Off(n1, 0)); err != nil {
			panic(err)
		}
	} else {
		Dlaset(Full, n2, k, zero, zero, q.Off(n1, 0))
	}

	Dlacpy(Full, n12, k, q, s.Matrix(n12, opts))
	if n12 != 0 {
		if err = goblas.Dgemm(NoTrans, NoTrans, n1, k, n12, one, q2.Matrix(n1, opts), s.Matrix(n12, opts), zero, q); err != nil {
			panic(err)
		}
	} else {
		Dlaset(Full, n1, k, zero, zero, q.Off(0, 0))
	}

	return
}
