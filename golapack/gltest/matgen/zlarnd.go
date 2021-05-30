package matgen

import (
	"math"
	"math/cmplx"
)

// Zlarnd returns a random complex number from a uniform or normal
// distribution.
func Zlarnd(idist *int, iseed *[]int) (zlarndReturn complex128) {
	var one, t1, t2, two, twopi, zero float64

	zero = 0.0
	one = 1.0
	two = 2.0
	twopi = 6.2831853071795864769252867663

	//     Generate a pair of real random numbers from a uniform (0,1)
	//     distribution
	t1 = Dlaran(iseed)
	t2 = Dlaran(iseed)

	if (*idist) == 1 {
		//        real and imaginary parts each uniform (0,1)
		zlarndReturn = complex(t1, t2)
	} else if (*idist) == 2 {
		//        real and imaginary parts each uniform (-1,1)
		zlarndReturn = complex(two*t1-one, two*t2-one)
	} else if (*idist) == 3 {
		//        real and imaginary parts each normal (0,1)
		zlarndReturn = complex(math.Sqrt(-two*math.Log(t1)), 0) * cmplx.Exp(complex(zero, twopi*t2))
	} else if (*idist) == 4 {
		//        uniform distribution on the unit disc abs(z) <= 1
		zlarndReturn = complex(math.Sqrt(t1), 0) * cmplx.Exp(complex(zero, twopi*t2))
	} else if (*idist) == 5 {
		//        uniform distribution on the unit circle abs(z) = 1
		zlarndReturn = cmplx.Exp(complex(zero, twopi*t2))
	}
	return
}
