package golapack

import "math"

// Dlapy2 returns sqrt(x**2+y**2), taking care not to cause unnecessary
// overflow.
func Dlapy2(x, y *float64) (dlapy2Return float64) {
	var xIsNan, yIsNan bool
	var one, w, xabs, yabs, z, zero float64

	zero = 0.0
	one = 1.0

	xIsNan = Disnan(int(*x))
	yIsNan = Disnan(int(*y))
	if xIsNan {
		dlapy2Return = (*x)
	}
	if yIsNan {
		dlapy2Return = (*y)
	}
	//
	if !(xIsNan || yIsNan) {
		xabs = math.Abs(*x)
		yabs = math.Abs(*y)
		w = maxf64(xabs, yabs)
		z = minf64(xabs, yabs)
		if z == zero {
			dlapy2Return = w
		} else {
			dlapy2Return = w * math.Sqrt(one+math.Pow(z/w, 2))
		}
	}
	return
}
