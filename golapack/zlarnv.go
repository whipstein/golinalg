package golapack

import (
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/mat"
)

// Zlarnv returns a vector of n random complex numbers from a uniform or
// normal distribution.
func Zlarnv(idist *int, iseed *[]int, n *int, x *mat.CVector) {
	var one, two, twopi, zero float64
	var i, il, iv, lv int

	u := vf(128)

	zero = 0.0
	one = 1.0
	two = 2.0
	lv = 128
	twopi = 6.2831853071795864769252867663

	for iv = 1; iv <= (*n); iv += lv / 2 {
		il = min(lv/2, (*n)-iv+1)

		//        Call DLARUV to generate 2*IL real numbers from a uniform (0,1)
		//        distribution (2*IL <= LV)
		Dlaruv(iseed, toPtr(2*il), u)

		if (*idist) == 1 {
			//           Copy generated numbers
			for i = 1; i <= il; i++ {
				x.Set(iv+i-1-1, complex(u.Get(2*i-1-1), u.Get(2*i-1)))
			}
		} else if (*idist) == 2 {
			//           Convert generated numbers to uniform (-1,1) distribution
			for i = 1; i <= il; i++ {
				x.Set(iv+i-1-1, complex(two*u.Get(2*i-1-1)-one, two*u.Get(2*i-1)-one))
			}
		} else if (*idist) == 3 {
			//           Convert generated numbers to normal (0,1) distribution
			for i = 1; i <= il; i++ {
				x.Set(iv+i-1-1, complex(math.Sqrt(-two*math.Log(u.Get(2*i-1-1))), 0)*cmplx.Exp(complex(zero, twopi*u.Get(2*i-1))))
			}
		} else if (*idist) == 4 {
			//           Convert generated numbers to complex numbers uniformly
			//           distributed on the unit disk
			for i = 1; i <= il; i++ {
				x.Set(iv+i-1-1, complex(math.Sqrt(u.Get(2*i-1-1)), 0)*cmplx.Exp(complex(zero, twopi*u.Get(2*i-1))))
			}
		} else if (*idist) == 5 {
			//           Convert generated numbers to complex numbers uniformly
			//           distributed on the unit circle
			for i = 1; i <= il; i++ {
				x.Set(iv+i-1-1, cmplx.Exp(complex(zero, twopi*u.Get(2*i-1))))
			}
		}
	}
}
