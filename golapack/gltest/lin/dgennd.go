package lin

import "github.com/whipstein/golinalg/mat"

// dgennd tests that its argument has a non-negative diagonal.
func dgennd(m, n int, a *mat.Matrix) (dgenndReturn bool) {
	var zero float64
	var i, k int

	zero = 0.0

	k = min(m, n)
	for i = 1; i <= k; i++ {
		if a.Get(i-1, i-1) < zero {
			dgenndReturn = false
			return
		}
	}
	dgenndReturn = true
	return
}
