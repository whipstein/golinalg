package eig

import "math"

// Zlctes returns .TRUE. if the eigenvalue Z/D is to be selected
// (specifically, in this subroutine, if the real part of the
// eigenvalue is negative), and otherwise it returns .FALSE..
//
// It is used by the test routine ZDRGES to test whether the driver
// routine ZGGES successfully sorts eigenvalues.
func Zlctes(z, d complex128) (zlctesReturn bool) {
	var czero complex128
	var one, zero, zmax float64

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)

	if d == czero {
		zlctesReturn = (real(z) < zero)
	} else {
		if real(z) == zero || real(d) == zero {
			zlctesReturn = (math.Copysign(one, imag(z)) != math.Copysign(one, imag(d)))
		} else if imag(z) == zero || imag(d) == zero {
			zlctesReturn = (math.Copysign(one, real(z)) != math.Copysign(one, real(d)))
		} else {
			zmax = maxf64(math.Abs(real(z)), math.Abs(imag(z)))
			zlctesReturn = ((real(z)/zmax)*real(d)+(imag(z)/zmax)*imag(d) < zero)
		}
	}

	return
}
