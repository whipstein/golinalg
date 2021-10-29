package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlarra Compute the splitting points with threshold SPLTOL.
// DLARRA sets any "small" off-diagonal elements to zero.
func Dlarra(n int, d, e, e2 *mat.Vector, spltol, tnrm float64, isplit *[]int) (nsplit int) {
	var eabs, tmp1, zero float64
	var i int

	zero = 0.0

	//     Quick return if possible
	if n <= 0 {
		return
	}

	//     Compute splitting points
	nsplit = 1
	if spltol < zero {
		//        Criterion based on absolute off-diagonal value
		tmp1 = math.Abs(spltol) * tnrm
		for i = 1; i <= n-1; i++ {
			eabs = math.Abs(e.Get(i - 1))
			if eabs <= tmp1 {
				e.Set(i-1, zero)
				e2.Set(i-1, zero)
				(*isplit)[nsplit-1] = i
				nsplit = nsplit + 1
			}
		}
	} else {
		//        Criterion that guarantees relative accuracy
		for i = 1; i <= n-1; i++ {
			eabs = math.Abs(e.Get(i - 1))
			if eabs <= spltol*math.Sqrt(math.Abs(d.Get(i-1)))*math.Sqrt(math.Abs(d.Get(i))) {
				e.Set(i-1, zero)
				e2.Set(i-1, zero)
				(*isplit)[nsplit-1] = i
				nsplit = nsplit + 1
			}
		}
	}
	(*isplit)[nsplit-1] = n

	return
}
