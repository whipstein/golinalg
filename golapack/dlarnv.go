package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlarnv returns a vector of n random real numbers from a uniform or
// normal distribution.
func Dlarnv(idist *int, iseed *[]int, n *int, x *mat.Vector) {
	var one, two, twopi float64
	var i, il, il2, iv, lv int

	u := vf(128)

	one = 1.0
	two = 2.0
	lv = 128
	twopi = 2 * math.Pi

	for iv = 1; iv <= (*n); iv += lv / 2 {
		il = min(lv/2, (*n)-iv+1)
		if (*idist) == 3 {
			il2 = 2 * il
		} else {
			il2 = il
		}

		//        Call DLARUV to generate IL2 numbers from a uniform (0,1)
		//        distribution (IL2 <= LV)
		Dlaruv(iseed, &il2, u)

		if (*idist) == 1 {
			//           Copy generated numbers

			for i = 1; i <= il; i++ {
				x.Set(iv+i-1-1, u.Get(i-1))
			}
		} else if (*idist) == 2 {
			//           Convert generated numbers to uniform (-1,1) distribution
			for i = 1; i <= il; i++ {
				x.Set(iv+i-1-1, two*u.Get(i-1)-one)
			}
		} else if (*idist) == 3 {
			//           Convert generated numbers to normal (0,1) distribution
			for i = 1; i <= il; i++ {
				x.Set(iv+i-1-1, math.Sqrt(-two*math.Log(u.Get(2*i-1-1)))*math.Cos(twopi*u.Get(2*i-1)))
			}
		}
	}
}
