package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Zlarfg generates a complex elementary reflector H of order n, such
// that
//
//       H**H * ( alpha ) = ( beta ),   H**H * H = I.
//              (   x   )   (   0  )
//
// where alpha and beta are scalars, with beta real, and x is an
// (n-1)-element complex vector. H is represented in the form
//
//       H = I - tau * ( 1 ) * ( 1 v**H ) ,
//                     ( v )
//
// where tau is a complex scalar and v is a complex (n-1)-element
// vector. Note that H is not hermitian.
//
// If the elements of x are all zero and alpha is real, then tau = 0
// and H is taken to be the unit matrix.
//
// Otherwise  1 <= real(tau) <= 2  and  abs(tau-1) <= 1 .
func Zlarfg(n int, alpha complex128, x *mat.CVector) (alphaOut, tau complex128) {
	var alphi, alphr, beta, one, rsafmn, safmin, xnorm, zero float64
	var j, knt int

	one = 1.0
	zero = 0.0
	alphaOut = alpha

	if n <= 0 {
		tau = complex(zero, 0)
		return
	}

	xnorm = goblas.Dznrm2(n-1, x)
	alphr = real(alphaOut)
	alphi = imag(alphaOut)

	if xnorm == zero && alphi == zero {
		//        H  =  I
		tau = complex(zero, 0)
	} else {
		//        general case
		beta = -math.Copysign(Dlapy3(alphr, alphi, xnorm), alphr)
		safmin = Dlamch(SafeMinimum) / Dlamch(Epsilon)
		rsafmn = one / safmin

		knt = 0
		if math.Abs(beta) < safmin {
			//           XNORM, BETA may be inaccurate; scale X and recompute them
		label10:
			;
			knt = knt + 1
			goblas.Zdscal(n-1, rsafmn, x)
			beta = beta * rsafmn
			alphi = alphi * rsafmn
			alphr = alphr * rsafmn
			if (math.Abs(beta) < safmin) && (knt < 20) {
				goto label10
			}

			//           New BETA is at most 1, at least SAFMIN
			xnorm = goblas.Dznrm2(n-1, x)
			alphaOut = complex(alphr, alphi)
			beta = -math.Copysign(Dlapy3(alphr, alphi, xnorm), alphr)
		}
		tau = complex((beta-alphr)/beta, -alphi/beta)
		alphaOut = Zladiv(complex(one, 0), alphaOut-complex(beta, 0))
		goblas.Zscal(n-1, alphaOut, x)

		//        If ALPHA is subnormal, it may lose relative accuracy
		for j = 1; j <= knt; j++ {
			beta = beta * safmin
		}
		alphaOut = complex(beta, 0)
	}

	return
}
