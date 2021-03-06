package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlarfgp generates a real elementary reflector H of order n, such
// that
//
//       H * ( alpha ) = ( beta ),   H**T * H = I.
//           (   x   )   (   0  )
//
// where alpha and beta are scalars, beta is non-negative, and x is
// an (n-1)-element real vector.  H is represented in the form
//
//       H = I - tau * ( 1 ) * ( 1 v**T ) ,
//                     ( v )
//
// where tau is a real scalar and v is a real (n-1)-element
// vector.
//
// If the elements of x are all zero, then tau = 0 and H is taken to be
// the unit matrix.
func Dlarfgp(n int, alpha float64, x *mat.Vector, incx int) (alphaOut, tau float64) {
	var beta, bignum, one, savealpha, smlnum, two, xnorm, zero float64
	var j, knt int

	two = 2.0
	one = 1.0
	zero = 0.0
	alphaOut = alpha

	if n <= 0 {
		tau = zero
		return
	}

	xnorm = x.Nrm2(n-1, incx)

	if xnorm == zero {
		//        H  =  [+/-1, 0; I], sign chosen so ALPHA >= 0
		if alphaOut >= zero {
			//           When TAU.eq.ZERO, the vector is special-cased to be
			//           all zeros in the application routines.  We do not need
			//           to clear it.
			tau = zero
		} else {
			//           However, the application routines rely on explicit
			//           zero checks when TAU.ne.ZERO, and we must clear X.
			tau = two
			for j = 1; j <= n-1; j++ {
				x.Set(1+(j-1)*incx-1, 0)
			}
			alphaOut = -alphaOut
		}
	} else {
		//        general case
		beta = math.Copysign(Dlapy2(alphaOut, xnorm), alphaOut)
		smlnum = Dlamch(SafeMinimum) / Dlamch(Epsilon)
		knt = 0
		if math.Abs(beta) < smlnum {
			//           XNORM, BETA may be inaccurate; scale X and recompute them
			bignum = one / smlnum
		label10:
			;
			knt = knt + 1
			x.Scal(n-1, bignum, incx)
			beta = beta * bignum
			alphaOut = alphaOut * bignum
			if (math.Abs(beta) < smlnum) && (knt < 20) {
				goto label10
			}

			//           New BETA is at most 1, at least SMLNUM
			xnorm = x.Nrm2(n-1, incx)
			beta = math.Copysign(Dlapy2(alphaOut, xnorm), alphaOut)
		}
		savealpha = alphaOut
		alphaOut = alphaOut + beta
		if beta < zero {
			beta = -beta
			tau = -alphaOut / beta
		} else {
			alphaOut = xnorm * (xnorm / alphaOut)
			tau = alphaOut / beta
			alphaOut = -alphaOut
		}

		if math.Abs(tau) <= smlnum {
			//           In the case where the computed TAU ends up being a denormalized number,
			//           it loses relative accuracy. This is a BIG problem. Solution: flush TAU
			//           to ZERO. This explains the next IF statement.
			//
			//           (Bug report provided by Pat Quillen from MathWorks on Jul 29, 2009.)
			//           (Thanks Pat. Thanks MathWorks.)
			if savealpha >= zero {
				tau = zero
			} else {
				tau = two
				for j = 1; j <= n-1; j++ {
					x.Set(1+(j-1)*incx-1, 0)
				}
				beta = -savealpha
			}

		} else {
			//           This is the general case.
			x.Scal(n-1, one/alphaOut, incx)

		}

		//        If BETA is subnormal, it may lose relative accuracy
		for j = 1; j <= knt; j++ {
			beta = beta * smlnum
		}
		alphaOut = beta
	}

	return
}
