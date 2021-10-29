package golapack

import "github.com/whipstein/golinalg/mat"

// Zlacgv conjugates a complex vector of length N.
func Zlacgv(n int, x *mat.CVector) {
	var i, ioff int

	if x.Inc == 1 {
		for i = 1; i <= n; i++ {
			x.Set(i-1, x.GetConj(i-1))
		}
	} else {
		ioff = 1
		if x.Inc < 0 {
			ioff = 1 - (n-1)*x.Inc
		}
		for i = 1; i <= n; i++ {
			x.Set(ioff-1, x.GetConj(ioff-1))
			ioff = ioff + x.Inc
		}
	}
}
