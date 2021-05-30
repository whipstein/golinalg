package golapack

import "math"

// Dlabad takes as input the values computed by DLAMCH for underflow and
// overflow, and returns the square root of each of these values if the
// log of LARGE is sufficiently large.  This subroutine is intended to
// identify machines with a large exponent range, such as the Crays, and
// redefine the underflow and overflow limits to be the square roots of
// the values computed by DLAMCH.  This subroutine is needed because
// DLAMCH does not compensate for poor arithmetic in the upper half of
// the exponent range, as is found on a Cray.
func Dlabad(small *float64, large *float64) {
	//     If it looks like we're on a Cray, take the square root of
	//     SMALL and LARGE to avoid overflow and underflow problems.
	if math.Log10(*large) > 2000. {
		(*small) = math.Sqrt(*small)
		(*large) = math.Sqrt(*large)
	}
}
