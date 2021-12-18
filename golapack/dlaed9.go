package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlaed9 finds the roots of the secular equation, as defined by the
// values in D, Z, and RHO, between KSTART and KSTOP.  It makes the
// appropriate calls to DLAED4 and then stores the new matrix of
// eigenvectors for use in calculating the next level of Z vectors.
func Dlaed9(k, kstart, kstop, n int, d *mat.Vector, q *mat.Matrix, rho float64, dlamda, w *mat.Vector, s *mat.Matrix) (info int, err error) {
	var temp float64
	var i, j int

	//     Test the input parameters.
	if k < 0 {
		err = fmt.Errorf("k < 0: k=%v", k)
	} else if kstart < 1 || kstart > max(1, k) {
		err = fmt.Errorf("kstart < 1 || kstart > max(1, k): k=%v, kstart=%v", k, kstart)
	} else if max(1, kstop) < kstart || kstop > max(1, k) {
		err = fmt.Errorf("max(1, kstop) < kstart || kstop > max(1, k): k=%v, kstart=%v, kstop=%v", k, kstart, kstop)
	} else if n < k {
		err = fmt.Errorf("n < k: k=%v, n=%v", k, n)
	} else if q.Rows < max(1, k) {
		err = fmt.Errorf("q.Rows < max(1, k): q.Rows=%v, k=%v", q.Rows, k)
	} else if s.Rows < max(1, k) {
		err = fmt.Errorf("s.Rows < max(1, k): s.Rows=%v, k=%v", s.Rows, k)
	}
	if err != nil {
		gltest.Xerbla2("Dlaed9", err)
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
	for i = 1; i <= n; i++ {
		dlamda.Set(i-1, Dlamc3(dlamda.GetPtr(i-1), dlamda.GetPtr(i-1))-dlamda.Get(i-1))
	}

	for j = kstart; j <= kstop; j++ {
		_d := d.GetPtr(j - 1)
		if *_d, info = Dlaed4(k, j, dlamda, w, q.Off(0, j-1).Vector(), rho); info != 0 {
			return
		}
	}

	if k == 1 || k == 2 {
		for i = 1; i <= k; i++ {
			for j = 1; j <= k; j++ {
				s.Set(j-1, i-1, q.Get(j-1, i-1))
			}
		}
		return
	}

	//     Compute updated W.
	s.OffIdx(0).Vector().Copy(k, w, 1, 1)

	//     Initialize W(I) = Q(I,I)
	w.Copy(k, q.OffIdx(0).Vector(), q.Rows+1, 1)
	for j = 1; j <= k; j++ {
		for i = 1; i <= j-1; i++ {
			w.Set(i-1, w.Get(i-1)*(q.Get(i-1, j-1)/(dlamda.Get(i-1)-dlamda.Get(j-1))))
		}
		for i = j + 1; i <= k; i++ {
			w.Set(i-1, w.Get(i-1)*(q.Get(i-1, j-1)/(dlamda.Get(i-1)-dlamda.Get(j-1))))
		}
	}
	for i = 1; i <= k; i++ {
		w.Set(i-1, math.Copysign(math.Sqrt(-w.Get(i-1)), s.Get(i-1, 0)))
	}

	//     Compute eigenvectors of the modified rank-1 modification.
	for j = 1; j <= k; j++ {
		for i = 1; i <= k; i++ {
			q.Set(i-1, j-1, w.Get(i-1)/q.Get(i-1, j-1))
		}
		temp = q.Off(0, j-1).Vector().Nrm2(k, 1)
		for i = 1; i <= k; i++ {
			s.Set(i-1, j-1, q.Get(i-1, j-1)/temp)
		}
	}

	return
}
