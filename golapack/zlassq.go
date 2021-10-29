package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Zlassq returns the values scl and ssq such that
//
//    ( scl**2 )*ssq = x( 1 )**2 +...+ x( n )**2 + ( scale**2 )*sumsq,
//
// where x( i ) = cmplx.Abs( X( 1 + ( i - 1 )*INCX ) ). The value of sumsq is
// assumed to be at least unity and the value of ssq will then satisfy
//
//    1.0 <= ssq <= ( sumsq + 2*n ).
//
// scale is assumed to be non-negative and scl returns the value
//
//    scl = max( scale, cmplx.Abs( real( x( i ) ) ), cmplx.Abs( aimag( x( i ) ) ) ),
//           i
//
// scale and sumsq must be supplied in SCALE and SUMSQ respectively.
// SCALE and SUMSQ are overwritten by scl and ssq respectively.
//
// The routine makes only one pass through the vector X.
func Zlassq(n int, x *mat.CVector, scale, sumsq float64) (scaleOut, sumsqOut float64) {
	var temp1, zero float64
	var ix int

	zero = 0.0
	scaleOut = scale
	sumsqOut = sumsq

	if n > 0 {
		for ix = 1; ix <= 1+(n-1)*x.Inc; ix += x.Inc {
			temp1 = math.Abs(real(x.Get(ix - 1)))
			if temp1 > zero || Disnan(int(temp1)) {
				if scaleOut < temp1 {
					sumsqOut = 1 + sumsqOut*math.Pow(scaleOut/temp1, 2)
					scaleOut = temp1
				} else {
					sumsqOut = sumsqOut + math.Pow(temp1/scaleOut, 2)
				}
			}
			temp1 = math.Abs(imag(x.Get(ix - 1)))
			if temp1 > zero || Disnan(int(temp1)) {
				if scaleOut < temp1 {
					sumsqOut = 1 + sumsqOut*math.Pow(scaleOut/temp1, 2)
					scaleOut = temp1
				} else {
					sumsqOut = sumsqOut + math.Pow(temp1/scaleOut, 2)
				}
			}
		}
	}

	return
}
