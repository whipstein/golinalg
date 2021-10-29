package golapack

import "github.com/whipstein/golinalg/mat"

// Ilazlc scans A for its last non-zero column.
func Ilazlc(m, n int, a *mat.CMatrix) (ilazlcReturn int) {
	var zero complex128
	var i int

	zero = (0.0 + 0.0*1i)

	//     Quick test for the common case where one corner is non-zero.
	if n == 0 {
		ilazlcReturn = n
	} else if a.Get(0, n-1) != zero || a.Get(m-1, n-1) != zero {
		ilazlcReturn = n
	} else {
		//     Now scan each column from the end, returning with the first non-zero.
		for ilazlcReturn = n; ilazlcReturn >= 1; ilazlcReturn-- {
			for i = 1; i <= m; i++ {
				if a.Get(i-1, ilazlcReturn-1) != zero {
					return
				}
			}
		}
	}
	return
}
