package golapack

import (
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Zlarfgp generates a complex elementary reflector H of order n, such
// that
//
//       H**H * ( alpha ) = ( beta ),   H**H * H = I.
//              (   x   )   (   0  )
//
// where alpha and beta are scalars, beta is real and non-negative, and
// x is an (n-1)-element complex vector.  H is represented in the form
//
//       H = I - tau * ( 1 ) * ( 1 v**H ) ,
//                     ( v )
//
// where tau is a complex scalar and v is a complex (n-1)-element
// vector. Note that H is not hermitian.
//
// If the elements of x are all zero and alpha is real, then tau = 0
// and H is taken to be the unit matrix.
func Zlarfgp(n *int, alpha *complex128, x *mat.CVector, incx *int, tau *complex128) {
	var savealpha complex128
	var alphi, alphr, beta, bignum, one, smlnum, two, xnorm, zero float64
	var j, knt int

	two = 2.0
	one = 1.0
	zero = 0.0

	if (*n) <= 0 {
		(*tau) = complex(zero, 0)
		return
	}

	xnorm = goblas.Dznrm2((*n)-1, x.Off(0, *incx))
	alphr = real(*alpha)
	alphi = imag(*alpha)

	if xnorm == zero {
		//        H  =  [1-alpha/math.Abs(alpha) 0; 0 I], sign chosen so ALPHA >= 0.
		if alphi == zero {
			if alphr >= zero {
				//              When TAU.eq.ZERO, the vector is special-cased to be
				//              all zeros in the application routines.  We do not need
				//              to clear it.
				(*tau) = complex(zero, 0)
			} else {
				//              However, the application routines rely on explicit
				//              zero checks when TAU.ne.ZERO, and we must clear X.
				(*tau) = complex(two, 0)
				for j = 1; j <= (*n)-1; j++ {
					x.SetRe(1+(j-1)*(*incx)-1, zero)
				}
				(*alpha) = -(*alpha)
			}
		} else {
			//           Only "reflecting" the diagonal entry to be real and non-negative.
			xnorm = Dlapy2(&alphr, &alphi)
			(*tau) = complex(one-alphr/xnorm, -alphi/xnorm)
			for j = 1; j <= (*n)-1; j++ {
				x.SetRe(1+(j-1)*(*incx)-1, zero)
			}
			(*alpha) = complex(xnorm, 0)
		}
	} else {
		//        general case
		beta = math.Copysign(Dlapy3(&alphr, &alphi, &xnorm), alphr)
		smlnum = Dlamch(SafeMinimum) / Dlamch(Epsilon)
		bignum = one / smlnum

		knt = 0
		if math.Abs(beta) < smlnum {
			//           XNORM, BETA may be inaccurate; scale X and recompute them
		label10:
			;
			knt = knt + 1
			goblas.Zdscal((*n)-1, bignum, x.Off(0, *incx))
			beta = beta * bignum
			alphi = alphi * bignum
			alphr = alphr * bignum
			if (math.Abs(beta) < smlnum) && (knt < 20) {
				goto label10
			}

			//           New BETA is at most 1, at least SMLNUM
			xnorm = goblas.Dznrm2((*n)-1, x.Off(0, *incx))
			(*alpha) = complex(alphr, alphi)
			beta = math.Copysign(Dlapy3(&alphr, &alphi, &xnorm), alphr)
		}
		savealpha = (*alpha)
		(*alpha) = (*alpha) + complex(beta, 0)
		if beta < zero {
			beta = -beta
			(*tau) = -(*alpha) / complex(beta, 0)
		} else {
			alphr = alphi * (alphi / real(*alpha))
			alphr = alphr + xnorm*(xnorm/real(*alpha))
			(*tau) = complex(alphr/beta, -alphi/beta)
			(*alpha) = complex(-alphr, alphi)
		}
		(*alpha) = Zladiv(toPtrc128(complex(one, 0)), alpha)

		if cmplx.Abs(*tau) <= smlnum {
			//           In the case where the computed TAU ends up being a denormalized number,
			//           it loses relative accuracy. This is a BIG problem. Solution: flush TAU
			//           to ZERO (or TWO or whatever makes a nonnegative real number for BETA).
			//
			//           (Bug report provided by Pat Quillen from MathWorks on Jul 29, 2009.)
			//           (Thanks Pat. Thanks MathWorks.)
			alphr = real(savealpha)
			alphi = imag(savealpha)
			if alphi == zero {
				if alphr >= zero {
					(*tau) = complex(zero, 0)
				} else {
					(*tau) = complex(two, 0)
					for j = 1; j <= (*n)-1; j++ {
						x.SetRe(1+(j-1)*(*incx)-1, zero)
					}
					beta = real(-savealpha)
				}
			} else {
				xnorm = Dlapy2(&alphr, &alphi)
				(*tau) = complex(one-alphr/xnorm, -alphi/xnorm)
				for j = 1; j <= (*n)-1; j++ {
					x.SetRe(1+(j-1)*(*incx)-1, zero)
				}
				beta = xnorm
			}

		} else {
			//           This is the general case.
			goblas.Zscal((*n)-1, *alpha, x.Off(0, *incx))

		}

		//        If BETA is subnormal, it may lose relative accuracy
		for j = 1; j <= knt; j++ {
			beta = beta * smlnum
		}
		(*alpha) = complex(beta, 0)
	}
}
