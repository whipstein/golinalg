package lin

import (
	"math"
	"strings"

	"github.com/whipstein/golinalg/golapack"
)

// zlatb5 sets parameters for the matrix generator based on the _type
// of matrix to be generated.
func zlatb5(path string, imat, n int) (_type byte, kl, ku int, anorm float64, mode int, cndnum float64, dist byte) {
	var first bool
	var badc1, badc2, eps, large, one, shrink, small, tenth, two float64

	shrink = 0.25
	tenth = 0.1
	one = 1.0
	two = 2.0

	first = true

	//     Set some constants for use in the subroutine.
	if first {
		first = false
		eps = golapack.Dlamch(Precision)
		badc2 = tenth / eps
		badc1 = math.Sqrt(badc2)
		small = golapack.Dlamch(SafeMinimum)
		large = one / small

		//        If it looks like we're on a Cray, take the square root of
		//        SMALL and LARGE to avoid overflow and underflow problems.
		small, large = golapack.Dlabad(small, large)
		small = shrink * (small / eps)
		large = one / small
	}

	c2 := path[1:3]

	//     Set some parameters
	dist = 'S'
	mode = 3

	//     Set TYPE, the _type of matrix to be generated.
	_type = strings.ToUpper(c2)[0]

	//     Set the lower and upper bandwidths.
	if imat == 1 {
		kl = 0
	} else {
		kl = max(n-1, 0)
	}
	ku = kl

	//     Set the condition number and norm.etc
	if imat == 3 {
		cndnum = 1.0e12
		mode = 2
	} else if imat == 4 {
		cndnum = 1.0e12
		mode = 1
	} else if imat == 5 {
		cndnum = 1.0e12
		mode = 3
	} else if imat == 6 {
		cndnum = badc1
	} else if imat == 7 {
		cndnum = badc2
	} else {
		cndnum = two
	}

	if imat == 8 {
		anorm = small
	} else if imat == 9 {
		anorm = large
	} else {
		anorm = one
	}

	if n <= 1 {
		cndnum = one
	}

	return
}
