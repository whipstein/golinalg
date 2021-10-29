package lin

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
)

// dget06 computes a test ratio to compare two values for RCOND.
func dget06(rcond, rcondc float64) (dget06Return float64) {
	var eps, one, zero float64

	zero = 0.0
	one = 1.0

	eps = golapack.Dlamch(Epsilon)
	if rcond > zero {
		if rcondc > zero {
			dget06Return = math.Max(rcond, rcondc)/math.Min(rcond, rcondc) - (one - eps)
		} else {
			dget06Return = rcond / eps
		}
	} else {
		if rcondc > zero {
			dget06Return = rcondc / eps
		} else {
			dget06Return = zero
		}
	}

	return
}
