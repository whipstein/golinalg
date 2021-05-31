package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
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
func Dlarfg(n *int, alpha *float64, x *mat.Vector, incx *int, tau *float64) {
	var beta, one, rsafmn, safmin, xnorm, zero float64
	var j, knt int

	one = 1.0
	zero = 0.0

	if (*n) <= 1 {
		(*tau) = zero
		return
	}

	xnorm = goblas.Dnrm2(toPtr((*n)-1), x, incx)

	if xnorm == zero {
		//        H  =  I
		(*tau) = zero
	} else {
		//        general case
		beta = -math.Copysign(Dlapy2(alpha, &xnorm), *alpha)
		safmin = Dlamch(SafeMinimum) / Dlamch(Epsilon)
		knt = 0
		if math.Abs(beta) < safmin {
			//           XNORM, BETA may be inaccurate; scale X and recompute them
			rsafmn = one / safmin
		label10:
			;
			knt = knt + 1
			goblas.Dscal(toPtr((*n)-1), &rsafmn, x, incx)
			beta = beta * rsafmn
			(*alpha) = (*alpha) * rsafmn
			if (math.Abs(beta) < safmin) && (knt < 20) {
				goto label10
			}

			//           New BETA is at most 1, at least SAFMIN
			xnorm = goblas.Dnrm2(toPtr((*n)-1), x, incx)
			beta = -math.Copysign(Dlapy2(alpha, &xnorm), *alpha)
		}
		(*tau) = (beta - (*alpha)) / beta
		goblas.Dscal(toPtr((*n)-1), toPtrf64(one/((*alpha)-beta)), x, incx)

		//        If ALPHA is subnormal, it may lose relative accuracy
		for j = 1; j <= knt; j++ {
			beta = beta * safmin
		}
		(*alpha) = beta
	}
}
