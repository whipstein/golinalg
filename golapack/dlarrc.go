package golapack

import (
	"golinalg/mat"
	"math"
)

// Dlarrc Find the number of eigenvalues of the symmetric tridiagonal matrix T
// that are in the interval (VL,VU] if JOBT = 'T', and of L D L^T
// if JOBT = 'L'.
func Dlarrc(jobt byte, n *int, vl, vu *float64, d, e *mat.Vector, pivmin *float64, eigcnt, lcnt, rcnt, info *int) {
	var matt bool
	var lpivot, rpivot, sl, su, tmp, tmp2, zero float64
	var i int

	zero = 0.0

	(*info) = 0

	//     Quick return if possible
	if (*n) <= 0 {
		return
	}

	(*lcnt) = 0
	(*rcnt) = 0
	(*eigcnt) = 0
	matt = jobt == 'T'
	if matt {
		//        Sturm sequence count on T
		lpivot = d.Get(0) - (*vl)
		rpivot = d.Get(0) - (*vu)
		if lpivot <= zero {
			(*lcnt) = (*lcnt) + 1
		}
		if rpivot <= zero {
			(*rcnt) = (*rcnt) + 1
		}
		for i = 1; i <= (*n)-1; i++ {
			tmp = math.Pow(e.Get(i-1), 2)
			lpivot = (d.Get(i+1-1) - (*vl)) - tmp/lpivot
			rpivot = (d.Get(i+1-1) - (*vu)) - tmp/rpivot
			if lpivot <= zero {
				(*lcnt) = (*lcnt) + 1
			}
			if rpivot <= zero {
				(*rcnt) = (*rcnt) + 1
			}
		}
	} else {
		//        Sturm sequence count on L D L^T
		sl = -(*vl)
		su = -(*vu)
		for i = 1; i <= (*n)-1; i++ {
			lpivot = d.Get(i-1) + sl
			rpivot = d.Get(i-1) + su
			if lpivot <= zero {
				(*lcnt) = (*lcnt) + 1
			}
			if rpivot <= zero {
				(*rcnt) = (*rcnt) + 1
			}
			tmp = e.Get(i-1) * d.Get(i-1) * e.Get(i-1)

			tmp2 = tmp / lpivot
			if tmp2 == zero {
				sl = tmp - (*vl)
			} else {
				sl = sl*tmp2 - (*vl)
			}

			tmp2 = tmp / rpivot
			if tmp2 == zero {
				su = tmp - (*vu)
			} else {
				su = su*tmp2 - (*vu)
			}
		}
		lpivot = d.Get((*n)-1) + sl
		rpivot = d.Get((*n)-1) + su
		if lpivot <= zero {
			(*lcnt) = (*lcnt) + 1
		}
		if rpivot <= zero {
			(*rcnt) = (*rcnt) + 1
		}
	}
	(*eigcnt) = (*rcnt) - (*lcnt)
}
