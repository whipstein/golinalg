package eig

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dort03 compares two orthogonal matrices U and V to see if their
// corresponding rows or columns span the same spaces.  The rows are
// checked if RC = 'R', and the columns are checked if RC = 'C'.
//
// RESULT is the maximum of
//
//    | V*V' - I | / ( MV ulp ), if RC = 'R', or
//
//    | V'*V - I | / ( MV ulp ), if RC = 'C',
//
// and the maximum over rows (or columns) 1 to K of
//
//    | U(i) - S*V(i) |/ ( N ulp )
//
// where S is +-1 (chosen to minimize the expression), U(i) is the i-th
// row (column) of U, and V(i) is the i-th row (column) of V.
func Dort03(rc byte, mu, mv, n, k *int, u *mat.Matrix, ldu *int, v *mat.Matrix, ldv *int, work *mat.Vector, lwork *int, result *float64, info *int) {
	var one, res1, res2, s, ulp, zero float64
	var i, irc, j, lmx int

	zero = 0.0
	one = 1.0

	//     Check inputs
	(*info) = 0
	if rc == 'R' {
		irc = 0
	} else if rc == 'C' {
		irc = 1
	} else {
		irc = -1
	}
	if irc == -1 {
		(*info) = -1
	} else if (*mu) < 0 {
		(*info) = -2
	} else if (*mv) < 0 {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*k) < 0 || (*k) > maxint(*mu, *mv) {
		(*info) = -5
	} else if (irc == 0 && (*ldu) < maxint(1, *mu)) || (irc == 1 && (*ldu) < maxint(1, *n)) {
		(*info) = -7
	} else if (irc == 0 && (*ldv) < maxint(1, *mv)) || (irc == 1 && (*ldv) < maxint(1, *n)) {
		(*info) = -9
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DORT03"), -(*info))
		return
	}

	//     Initialize result
	(*result) = zero
	if (*mu) == 0 || (*mv) == 0 || (*n) == 0 {
		return
	}

	//     Machine constants
	ulp = golapack.Dlamch(Precision)

	if irc == 0 {
		//        Compare rows
		res1 = zero
		for i = 1; i <= (*k); i++ {
			lmx = goblas.Idamax(n, u.Vector(i-1, 0), ldu)
			s = math.Copysign(one, u.Get(i-1, lmx-1)) * math.Copysign(one, v.Get(i-1, lmx-1))
			for j = 1; j <= (*n); j++ {
				res1 = maxf64(res1, math.Abs(u.Get(i-1, j-1)-s*v.Get(i-1, j-1)))
			}
		}
		res1 = res1 / (float64(*n) * ulp)

		//        Compute orthogonality of rows of V.
		Dort01('R', mv, n, v, ldv, work, lwork, &res2)

	} else {
		//        Compare columns
		res1 = zero
		for i = 1; i <= (*k); i++ {
			lmx = goblas.Idamax(n, u.Vector(0, i-1), toPtr(1))
			s = math.Copysign(one, u.Get(lmx-1, i-1)) * math.Copysign(one, v.Get(lmx-1, i-1))
			for j = 1; j <= (*n); j++ {
				res1 = maxf64(res1, math.Abs(u.Get(j-1, i-1)-s*v.Get(j-1, i-1)))
			}
		}
		res1 = res1 / (float64(*n) * ulp)

		//        Compute orthogonality of columns of V.
		Dort01('C', n, mv, v, ldv, work, lwork, &res2)
	}

	(*result) = minf64(maxf64(res1, res2), one/ulp)
}
