package golapack

import "github.com/whipstein/golinalg/mat"

// Iladlr scans A for its last non-zero row.
func Iladlr(m, n *int, a *mat.Matrix, lda *int) (iladlrReturn int) {
	var zero float64
	var i, j int

	zero = 0.0

	//     Quick test for the common case where one corner is non-zero.
	if (*m) == 0 {
		iladlrReturn = (*m)
	} else if a.Get((*m)-1, 0) != zero || a.Get((*m)-1, (*n)-1) != zero {
		iladlrReturn = (*m)
	} else {
		//     Scan up each column tracking the last zero row seen.
		iladlrReturn = 0
		for j = 1; j <= (*n); j++ {
			i = (*m)
			for (a.Get(maxint(i, 1)-1, j-1) == zero) && (i >= 1) {
				i = i - 1
			}
			iladlrReturn = maxint(iladlrReturn, i)
		}
	}
	return
}
