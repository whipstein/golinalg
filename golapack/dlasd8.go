package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlasd8 finds the square roots of the roots of the secular equation,
// as defined by the values in DSIGMA and Z. It makes the appropriate
// calls to DLASD4, and stores, for each  element in D, the distance
// to its two nearest poles (elements in DSIGMA). It also updates
// the arrays VF and VL, the first and last components of all the
// right singular vectors of the original bidiagonal matrix.
//
// Dlasd8 is called from DLASD6.
func Dlasd8(icompq, k int, d, z, vf, vl, difl *mat.Vector, difr *mat.Matrix, dsigma, work *mat.Vector) (info int, err error) {
	var diflj, difrj, dj, dsigj, dsigjp, one, rho, temp float64
	var i, iwk1, iwk2, iwk2i, iwk3, iwk3i, j int

	one = 1.0

	//     Test the input parameters.
	if (icompq < 0) || (icompq > 1) {
		err = fmt.Errorf("(icompq < 0) || (icompq > 1): icompq=%v", icompq)
	} else if k < 1 {
		err = fmt.Errorf("k < 1: k=%v", k)
	} else if difr.Rows < k {
		err = fmt.Errorf("difr.Rows < k: difr.Rows=%v, k=%v", difr.Rows, k)
	}
	if err != nil {
		gltest.Xerbla2("Dlasd8", err)
		return
	}

	//     Quick return if possible
	if k == 1 {
		d.Set(0, math.Abs(z.Get(0)))
		difl.Set(0, d.Get(0))
		if icompq == 1 {
			difl.Set(1, one)
			difr.Set(0, 1, one)
		}
		return
	}

	//     Modify values DSIGMA(i) to make sure all DSIGMA(i)-DSIGMA(j) can
	//     be computed with high relative accuracy (barring over/underflow).
	//     This is a problem on machines without a guard digit in
	//     add/subtract (Cray XMP, Cray YMP, Cray C 90 and Cray 2).
	//     The following code replaces DSIGMA(I) by 2*DSIGMA(I)-DSIGMA(I),
	//     which on any of these machines zeros out the bottommost
	//     bit of DSIGMA(I) if it is 1; this makes the subsequent
	//     subtractions DSIGMA(I)-DSIGMA(J) unproblematic when cancellation
	//     occurs. On binary machines with a guard digit (almost all
	//     machines) it does not change DSIGMA(I) at all. On hexadecimal
	//     and decimal machines with a guard digit, it slightly
	//     changes the bottommost bits of DSIGMA(I). It does not account
	//     for hexadecimal or decimal machines without guard digits
	//     (we know of none). We use a subroutine call to compute
	//     2*DLAMBDA(I) to prevent optimizing compilers from eliminating
	//     this code.
	for i = 1; i <= k; i++ {
		dsigma.Set(i-1, Dlamc3(dsigma.GetPtr(i-1), dsigma.GetPtr(i-1))-dsigma.Get(i-1))
	}

	//     Book keeping.
	iwk1 = 1
	iwk2 = iwk1 + k
	iwk3 = iwk2 + k
	iwk2i = iwk2 - 1
	iwk3i = iwk3 - 1

	//     Normalize Z.
	rho = goblas.Dnrm2(k, z.Off(0, 1))
	if err = Dlascl('G', 0, 0, rho, one, k, 1, z.Matrix(k, opts)); err != nil {
		panic(err)
	}
	rho = rho * rho

	//     Initialize WORK(IWK3).
	Dlaset(Full, k, 1, one, one, work.MatrixOff(iwk3-1, k, opts))

	//     Compute the updated singular values, the arrays DIFL, DIFR,
	//     and the updated Z.
	for j = 1; j <= k; j++ {
		*d.GetPtr(j - 1), info = Dlasd4(k, j, dsigma, z, work.Off(iwk1-1), rho, work.Off(iwk2-1))

		//        If the root finder fails, report the convergence failure.
		if info != 0 {
			return
		}
		work.Set(iwk3i+j-1, work.Get(iwk3i+j-1)*work.Get(j-1)*work.Get(iwk2i+j-1))
		difl.Set(j-1, -work.Get(j-1))
		difr.Set(j-1, 0, -work.Get(j))
		for i = 1; i <= j-1; i++ {
			work.Set(iwk3i+i-1, work.Get(iwk3i+i-1)*work.Get(i-1)*work.Get(iwk2i+i-1)/(dsigma.Get(i-1)-dsigma.Get(j-1))/(dsigma.Get(i-1)+dsigma.Get(j-1)))
		}
		for i = j + 1; i <= k; i++ {
			work.Set(iwk3i+i-1, work.Get(iwk3i+i-1)*work.Get(i-1)*work.Get(iwk2i+i-1)/(dsigma.Get(i-1)-dsigma.Get(j-1))/(dsigma.Get(i-1)+dsigma.Get(j-1)))
		}
	}

	//     Compute updated Z.
	for i = 1; i <= k; i++ {
		z.Set(i-1, math.Copysign(math.Sqrt(math.Abs(work.Get(iwk3i+i-1))), z.Get(i-1)))
	}

	//     Update VF and VL.
	for j = 1; j <= k; j++ {
		diflj = difl.Get(j - 1)
		dj = d.Get(j - 1)
		dsigj = -dsigma.Get(j - 1)
		if j < k {
			difrj = -difr.Get(j-1, 0)
			dsigjp = -dsigma.Get(j + 1 - 1)
		}
		work.Set(j-1, -z.Get(j-1)/diflj/(dsigma.Get(j-1)+dj))
		for i = 1; i <= j-1; i++ {
			work.Set(i-1, z.Get(i-1)/(Dlamc3(dsigma.GetPtr(i-1), &dsigj)-diflj)/(dsigma.Get(i-1)+dj))
		}
		for i = j + 1; i <= k; i++ {
			work.Set(i-1, z.Get(i-1)/(Dlamc3(dsigma.GetPtr(i-1), &dsigjp)+difrj)/(dsigma.Get(i-1)+dj))
		}
		temp = goblas.Dnrm2(k, work.Off(0, 1))
		work.Set(iwk2i+j-1, goblas.Ddot(k, work.Off(0, 1), vf.Off(0, 1))/temp)
		work.Set(iwk3i+j-1, goblas.Ddot(k, work.Off(0, 1), vl.Off(0, 1))/temp)
		if icompq == 1 {
			difr.Set(j-1, 1, temp)
		}
	}

	goblas.Dcopy(k, work.Off(iwk2-1, 1), vf.Off(0, 1))
	goblas.Dcopy(k, work.Off(iwk3-1, 1), vl.Off(0, 1))

	return
}
