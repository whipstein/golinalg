package golapack

import "golinalg/mat"

// Zlacgv conjugates a complex vector of length N.
func Zlacgv(n *int, x *mat.CVector, incx *int) {
	var i, ioff int

	if (*incx) == 1 {
		for i = 1; i <= (*n); i++ {
			x.Set(i-1, x.GetConj(i-1))
		}
	} else {
		ioff = 1
		if (*incx) < 0 {
			ioff = 1 - ((*n)-1)*(*incx)
		}
		for i = 1; i <= (*n); i++ {
			x.Set(ioff-1, x.GetConj(ioff-1))
			ioff = ioff + (*incx)
		}
	}
}
