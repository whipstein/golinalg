package lin

import "golinalg/mat"

// Zgennd tests that its argument has a real, non-negative diagonal.
func Zgennd(m, n *int, a *mat.CMatrix, lda *int) (zgenndReturn bool) {
	var aii complex128
	var zero float64
	var i, k int

	zero = 0.0

	k = minint(*m, *n)
	for i = 1; i <= k; i++ {
		aii = a.Get(i-1, i-1)
		if real(aii) < zero || imag(aii) != zero {
			zgenndReturn = false
			return
		}
	}
	zgenndReturn = true
	return
}