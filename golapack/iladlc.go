package golapack

import "github.com/whipstein/golinalg/mat"

// Iladlc scans A for its last non-zero column.
func Iladlc(m, n int, a *mat.Matrix) (iladlcReturn int) {
	var zero float64
	var i int

	zero = 0.0

	//     Quick test for the common case where one corner is non-zero.
	if n == 0 {
		iladlcReturn = n
	} else if a.Get(0, n-1) != zero || a.Get(m-1, n-1) != zero {
		iladlcReturn = n
	} else {
		//     Now scan each column from the end, returning with the first non-zero.
		for iladlcReturn = n; iladlcReturn >= 1; iladlcReturn-- {
			for i = 1; i <= m; i++ {
				if a.Get(i-1, iladlcReturn-1) != zero {
					return
				}
			}
		}
	}
	return
}
