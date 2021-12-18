package lin

import (
	"github.com/whipstein/golinalg/mat"
)

// dlaord sorts the elements of a vector x in increasing or decreasing
// order.
func dlaord(job byte, n int, x *mat.Vector, incx int) {
	var temp float64
	var i, inc, ix, ixnext int

	inc = abs(incx)
	if job == 'I' {
		//        Sort in increasing order
		for i = 2; i <= n; i++ {
			ix = 1 + (i-1)*inc
		label10:
			;
			if ix == 1 {
				goto label20
			}
			ixnext = ix - inc
			if x.Get(ix-1) > x.Get(ixnext-1) {
				goto label20
			} else {
				temp = x.Get(ix - 1)
				x.Set(ix-1, x.Get(ixnext-1))
				x.Set(ixnext-1, temp)
			}
			ix = ixnext
			goto label10
		label20:
		}

	} else if job == 'D' {
		//        Sort in decreasing order
		for i = 2; i <= n; i++ {
			ix = 1 + (i-1)*inc
		label30:
			;
			if ix == 1 {
				goto label40
			}
			ixnext = ix - inc
			if x.Get(ix-1) < x.Get(ixnext-1) {
				goto label40
			} else {
				temp = x.Get(ix - 1)
				x.Set(ix-1, x.Get(ixnext-1))
				x.Set(ixnext-1, temp)
			}
			ix = ixnext
			goto label30
		label40:
		}
	}
}
