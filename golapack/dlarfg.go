package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlarfg generates a real elementary reflector H of order n, such
// that
//
//       H * ( alpha ) = ( beta ),   H**T * H = I.
//           (   x   )   (   0  )
//
// where alpha and beta are scalars, and x is an (n-1)-element real
// vector. H is represented in the form
//
//       H = I - tau * ( 1 ) * ( 1 v**T ) ,
//                     ( v )
//
// where tau is a real scalar and v is a real (n-1)-element
// vector.
//
// If the elements of x are all zero, then tau = 0 and H is taken to be
// the unit matrix.
//
// Otherwise  1 <= tau <= 2.
func Dlarfg(n int, alpha float64, x *mat.Vector, incx int) (alphaOut, tau float64) {
	var beta, one, rsafmn, safmin, xnorm, zero float64
	var j, knt int

	one = 1.0
	zero = 0.0
	alphaOut = alpha

	if n <= 1 {
		tau = zero
		return
	}

	xnorm = x.Nrm2(n-1, incx)

	if xnorm == zero {
		//        H  =  I
		tau = zero
	} else {
		//        general case
		beta = -math.Copysign(Dlapy2(alphaOut, xnorm), alphaOut)
		safmin = Dlamch(SafeMinimum) / Dlamch(Epsilon)
		knt = 0
		if math.Abs(beta) < safmin {
			//           XNORM, BETA may be inaccurate; scale X and recompute them
			rsafmn = one / safmin
		label10:
			;
			knt = knt + 1
			x.Scal(n-1, rsafmn, incx)
			beta = beta * rsafmn
			alphaOut = alphaOut * rsafmn
			if (math.Abs(beta) < safmin) && (knt < 20) {
				goto label10
			}

			//           New BETA is at most 1, at least SAFMIN
			xnorm = x.Nrm2(n-1, incx)
			beta = -math.Copysign(Dlapy2(alphaOut, xnorm), alphaOut)
		}
		tau = (beta - alphaOut) / beta
		x.Scal(n-1, one/(alphaOut-beta), incx)

		//        If ALPHA is subnormal, it may lose relative accuracy
		for j = 1; j <= knt; j++ {
			beta = beta * safmin
		}
		alphaOut = beta
	}

	return
}
