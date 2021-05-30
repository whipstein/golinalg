package golapack

import (
	"golinalg/mat"
	"math"
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
func Zlassq(n *int, x *mat.CVector, incx *int, scale, sumsq *float64) {
	var temp1, zero float64
	var ix int

	zero = 0.0

	if (*n) > 0 {
		for ix = 1; ix <= 1+((*n)-1)*(*incx); ix += (*incx) {
			temp1 = math.Abs(real(x.Get(ix - 1)))
			if temp1 > zero || Disnan(int(temp1)) {
				if (*scale) < temp1 {
					(*sumsq) = 1 + (*sumsq)*math.Pow((*scale)/temp1, 2)
					(*scale) = temp1
				} else {
					(*sumsq) = (*sumsq) + math.Pow(temp1/(*scale), 2)
				}
			}
			temp1 = math.Abs(imag(x.Get(ix - 1)))
			if temp1 > zero || Disnan(int(temp1)) {
				if (*scale) < temp1 {
					(*sumsq) = 1 + (*sumsq)*math.Pow((*scale)/temp1, 2)
					(*scale) = temp1
				} else {
					(*sumsq) = (*sumsq) + math.Pow(temp1/(*scale), 2)
				}
			}
		}
	}
}
