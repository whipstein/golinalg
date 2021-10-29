package eig

import "math"

// dlctes returns .TRUE. if the eigenvalue (ZR/D) + sqrt(-1)*(ZI/D)
// is to be selected (specifically, in this subroutine, if the real
// part of the eigenvalue is negative), and otherwise it returns
// .FALSE..
//
// It is used by the test routine DDRGES to test whether the driver
// routine DGGES successfully sorts eigenvalues.
func dlctes(zr, zi, d *float64) (dlctesReturn bool) {
	var one, zero float64

	zero = 0.0
	one = 1.0

	if *d == zero {
		dlctesReturn = (*zr < zero)
	} else {
		dlctesReturn = (math.Copysign(one, *zr) != math.Copysign(one, *d))
	}

	return
}
