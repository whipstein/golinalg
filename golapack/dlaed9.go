package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlaed9 finds the roots of the secular equation, as defined by the
// values in D, Z, and RHO, between KSTART and KSTOP.  It makes the
// appropriate calls to DLAED4 and then stores the new matrix of
// eigenvectors for use in calculating the next level of Z vectors.
func Dlaed9(k, kstart, kstop, n *int, d *mat.Vector, q *mat.Matrix, ldq *int, rho *float64, dlamda, w *mat.Vector, s *mat.Matrix, lds *int, info *int) {
	var temp float64
	var i, j int

	//     Test the input parameters.
	(*info) = 0

	if (*k) < 0 {
		(*info) = -1
	} else if (*kstart) < 1 || (*kstart) > maxint(1, *k) {
		(*info) = -2
	} else if maxint(1, *kstop) < (*kstart) || (*kstop) > maxint(1, *k) {
		(*info) = -3
	} else if (*n) < (*k) {
		(*info) = -4
	} else if (*ldq) < maxint(1, *k) {
		(*info) = -7
	} else if (*lds) < maxint(1, *k) {
		(*info) = -12
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLAED9"), -(*info))
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
	for i = 1; i <= (*n); i++ {
		dlamda.Set(i-1, Dlamc3(dlamda.GetPtr(i-1), dlamda.GetPtr(i-1))-dlamda.Get(i-1))
	}

	for j = (*kstart); j <= (*kstop); j++ {
		Dlaed4(k, &j, dlamda, w, q.Vector(0, j-1), rho, d.GetPtr(j-1), info)

		//        If the zero finder fails, the computation is terminated.
		if (*info) != 0 {
			return
		}
	}

	if (*k) == 1 || (*k) == 2 {
		for i = 1; i <= (*k); i++ {
			for j = 1; j <= (*k); j++ {
				s.Set(j-1, i-1, q.Get(j-1, i-1))
			}
		}
		return
	}

	//     Compute updated W.
	goblas.Dcopy(*k, w, 1, s.VectorIdx(0), 1)

	//     Initialize W(I) = Q(I,I)
	goblas.Dcopy(*k, q.VectorIdx(0), (*ldq)+1, w, 1)
	for j = 1; j <= (*k); j++ {
		for i = 1; i <= j-1; i++ {
			w.Set(i-1, w.Get(i-1)*(q.Get(i-1, j-1)/(dlamda.Get(i-1)-dlamda.Get(j-1))))
		}
		for i = j + 1; i <= (*k); i++ {
			w.Set(i-1, w.Get(i-1)*(q.Get(i-1, j-1)/(dlamda.Get(i-1)-dlamda.Get(j-1))))
		}
	}
	for i = 1; i <= (*k); i++ {
		w.Set(i-1, signf64(math.Sqrt(-w.Get(i-1)), s.Get(i-1, 0)))
	}

	//     Compute eigenvectors of the modified rank-1 modification.
	for j = 1; j <= (*k); j++ {
		for i = 1; i <= (*k); i++ {
			q.Set(i-1, j-1, w.Get(i-1)/q.Get(i-1, j-1))
		}
		temp = goblas.Dnrm2(*k, q.Vector(0, j-1), 1)
		for i = 1; i <= (*k); i++ {
			s.Set(i-1, j-1, q.Get(i-1, j-1)/temp)
		}
	}
}
