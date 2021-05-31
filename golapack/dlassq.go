package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlassq returns the values  scl  and  smsq  such that
//
//    ( scl**2 )*smsq = x( 1 )**2 +...+ x( n )**2 + ( scale**2 )*sumsq,
//
// where  x( i ) = X( 1 + ( i - 1 )*INCX ). The value of  sumsq  is
// assumed to be non-negative and  scl  returns the value
//
//    scl = max( scale, abs( x( i ) ) ).
//
// scale and sumsq must be supplied in SCALE and SUMSQ and
// scl and smsq are overwritten on SCALE and SUMSQ respectively.
//
// The routine makes only one pass through the vector x.
func Dlassq(n *int, x *mat.Vector, incx *int, scale, sumsq *float64) {
	var absxi, zero float64
	var ix int

	zero = 0.0

	if (*n) > 0 {
		for ix = 1; ix <= 1+((*n)-1)*(*incx); ix += (*incx) {
			absxi = math.Abs(x.Get(ix - 1))
			if absxi > zero || Disnan(int(absxi)) {
				if (*scale) < absxi {
					(*sumsq) = 1 + (*sumsq)*math.Pow((*scale)/absxi, 2)
					(*scale) = absxi
				} else {
					(*sumsq) = (*sumsq) + math.Pow(absxi/(*scale), 2)
				}
			}
		}
	}
}
