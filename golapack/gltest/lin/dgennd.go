package lin

import "golinalg/mat"

// Dgennd tests that its argument has a non-negative diagonal.
func Dgennd(m, n *int, a *mat.Matrix, lda *int) (dgenndReturn bool) {
	var zero float64
	var i, k int

	zero = 0.0

	k = minint(*m, *n)
	for i = 1; i <= k; i++ {
		if a.Get(i-1, i-1) < zero {
			dgenndReturn = false
			return
		}
	}
	dgenndReturn = true
	return
}
