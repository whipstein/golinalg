package lin

import "github.com/whipstein/golinalg/mat"

// zgennd tests that its argument has a real, non-negative diagonal.
func zgennd(m, n int, a *mat.CMatrix) (zgenndReturn bool) {
	var aii complex128
	var zero float64
	var i, k int

	zero = 0.0

	k = min(m, n)
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
