package matgen

import "math"

// Dlarnd returns a random real number from a uniform or normal
// distribution.
func Dlarnd(idist *int, iseed *[]int) (dlarndReturn float64) {
	var one, t1, t2, two, twopi float64

	one = 1.0
	two = 2.0
	twopi = 2 * math.Pi

	//     Generate a real random number from a uniform (0,1) distribution
	t1 = Dlaran(iseed)

	if (*idist) == 1 {
		//        uniform (0,1)
		dlarndReturn = t1
	} else if (*idist) == 2 {
		//        uniform (-1,1)
		dlarndReturn = two*t1 - one
	} else if (*idist) == 3 {
		//        normal (0,1)
		t2 = Dlaran(iseed)
		dlarndReturn = math.Sqrt(-two*math.Log(t1)) * math.Cos(twopi*t2)
	}
	return
}
