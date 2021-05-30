package golapack

import "golinalg/mat"

// Ilazlr scans A for its last non-zero row.
func Ilazlr(m, n *int, a *mat.CMatrix, lda *int) (ilazlrReturn int) {
	var zero complex128
	var i, j int

	zero = (0.0 + 0.0*1i)

	//     Quick test for the common case where one corner is non-zero.
	if (*m) == 0 {
		ilazlrReturn = (*m)
	} else if a.Get((*m)-1, 0) != zero || a.Get((*m)-1, (*n)-1) != zero {
		ilazlrReturn = (*m)
	} else {
		//     Scan up each column tracking the last zero row seen.
		ilazlrReturn = 0
		for j = 1; j <= (*n); j++ {
			i = (*m)
			for (a.Get(maxint(i, 1)-1, j-1) == zero) && (i >= 1) {
				i = i - 1
			}
			ilazlrReturn = maxint(ilazlrReturn, i)
		}
	}
	return
}
