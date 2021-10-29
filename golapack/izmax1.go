package golapack

import (
	"github.com/whipstein/golinalg/mat"
)

// Izmax1 finds the index of the first vector element of maximum absolute value.
//
// Based on IZAMAX from Level 1 BLAS.
// The change is to use the 'genuine' absolute value.
func Izmax1(n int, zx *mat.CVector) (izmax1Return int) {
	var dmax float64
	var i, ix int

	izmax1Return = 0
	if n < 1 || zx.Inc <= 0 {
		return
	}
	izmax1Return = 1
	if n == 1 {
		return
	}
	if zx.Inc == 1 {
		//        code for increment equal to 1
		dmax = zx.GetMag(0)
		for i = 2; i <= n; i++ {
			if zx.GetMag(i-1) > dmax {
				izmax1Return = i
				dmax = zx.GetMag(i - 1)
			}
		}
	} else {
		//        code for increment not equal to 1
		ix = 1
		dmax = zx.GetMag(0)
		ix = ix + zx.Inc
		for i = 2; i <= n; i++ {
			if zx.GetMag(ix-1) > dmax {
				izmax1Return = i
				dmax = zx.GetMag(ix - 1)
			}
			ix = ix + zx.Inc
		}
	}
	return
}
