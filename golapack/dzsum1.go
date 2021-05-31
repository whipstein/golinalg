package golapack

import (
	"github.com/whipstein/golinalg/mat"
)

// Dzsum1 takes the sum of the absolute values of a complex
// vector and returns a double precision result.
//
// Based on DZASUM from the Level 1 BLAS.
// The change is to use the 'genuine' absolute value.
func Dzsum1(n *int, cx *mat.CVector, incx *int) (dzsum1Return float64) {
	var stemp float64
	var i, nincx int

	dzsum1Return = 0.0
	stemp = 0.0
	if (*n) <= 0 {
		return
	}
	if (*incx) == 1 {
		goto label20
	}

	//     CODE FOR INCREMENT NOT EQUAL TO 1
	nincx = (*n) * (*incx)
	for i = 1; i <= nincx; i += (*incx) {
		//        NEXT LINE MODIFIED.
		stemp = stemp + cx.GetMag(i-1)
	}
	dzsum1Return = stemp
	return

	//     CODE FOR INCREMENT EQUAL TO 1
label20:
	;
	for i = 1; i <= (*n); i++ {
		//        NEXT LINE MODIFIED.
		stemp = stemp + cx.GetMag(i-1)
	}
	dzsum1Return = stemp
	return
}
