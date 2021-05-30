package golapack

import "math"

// Dlartgs generates a plane rotation designed to introduce a bulge in
// Golub-Reinsch-style implicit QR iteration for the bidiagonal SVD
// problem. X and Y are the top-row entries, and SIGMA is the shift.
// The computed CS and SN define a plane rotation satisfying
//
//    [  CS  SN  ]  .  [ X^2 - SIGMA ]  =  [ R ],
//    [ -SN  CS  ]     [    X * Y    ]     [ 0 ]
//
// with R nonnegative.  If X^2 - SIGMA and X * Y are 0, then the
// rotation is by PI/2.
func Dlartgs(x, y, sigma, cs, sn *float64) {
	var negone, one, r, s, thresh, w, z, zero float64

	negone = -1.0
	one = 1.0
	zero = 0.0

	thresh = Dlamch(Epsilon)

	//     Compute the first column of B**T*B - SIGMA^2*I, up to a scale
	//     factor.
	if ((*sigma) == zero && math.Abs(*x) < thresh) || (math.Abs(*x) == (*sigma) && (*y) == zero) {
		z = zero
		w = zero
	} else if (*sigma) == zero {
		if (*x) >= zero {
			z = (*x)
			w = (*y)
		} else {
			z = -(*x)
			w = -(*y)
		}
	} else if math.Abs(*x) < thresh {
		z = -(*sigma) * (*sigma)
		w = zero
	} else {
		if (*x) >= zero {
			s = one
		} else {
			s = negone
		}
		z = s * (math.Abs(*x) - (*sigma)) * (s + (*sigma)/(*x))
		w = s * (*y)
	}

	//     Generate the rotation.
	//     CALL DLARTGP( Z, W, CS, SN, R ) might seem more natural;
	//     reordering the arguments ensures that if Z = 0 then the rotation
	//     is by PI/2.
	Dlartgp(&w, &z, sn, cs, &r)
}
