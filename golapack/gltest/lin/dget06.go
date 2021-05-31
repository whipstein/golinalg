package lin

import "github.com/whipstein/golinalg/golapack"

// Dget06 computes a test ratio to compare two values for RCOND.
func Dget06(rcond *float64, rcondc *float64) (dget06Return float64) {
	var eps, one, rat, zero float64

	zero = 0.0
	one = 1.0

	eps = golapack.Dlamch(Epsilon)
	if (*rcond) > zero {
		if (*rcondc) > zero {
			rat = maxf64(*rcond, *rcondc)/minf64(*rcond, *rcondc) - (one - eps)
		} else {
			rat = (*rcond) / eps
		}
	} else {
		if (*rcondc) > zero {
			rat = (*rcondc) / eps
		} else {
			rat = zero
		}
	}
	dget06Return = rat
	return
}
