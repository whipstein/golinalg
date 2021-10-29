package golapack

import "math"

// Dlapy3 returns math.Sqrt(x**2+y**2+z**2), taking care not to cause
// unnecessary overflow.
func Dlapy3(x, y, z float64) (dlapy3Return float64) {
	var w, xabs, yabs, zabs, zero float64

	zero = 0.0

	xabs = math.Abs(x)
	yabs = math.Abs(y)
	zabs = math.Abs(z)
	w = math.Max(xabs, math.Max(yabs, zabs))
	if w == zero {
		//     W can be zero for max(0,nan,0)
		//     adding all three entries together will make sure
		//     NaN will not disappear.
		dlapy3Return = xabs + yabs + zabs
	} else {
		dlapy3Return = w * math.Sqrt(math.Pow(xabs/w, 2)+math.Pow(yabs/w, 2)+math.Pow(zabs/w, 2))
	}
	return
}
