package golapack

import (
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/mat"
)

// Zlargv generates a vector of complex plane rotations with real
// cosines, determined by elements of the complex vectors x and y.
// For i = 1,2,...,n
//
//    (        c(i)   s(i) ) ( x(i) ) = ( r(i) )
//    ( -conjg(s(i))  c(i) ) ( y(i) ) = (   0  )
//
//    where c(i)**2 + ABS(s(i))**2 = 1
//
// The following conventions are used (these are the same as in ZLARTG,
// but differ from the BLAS1 routine ZROTG):
//    If y(i)=0, then c(i)=1 and s(i)=0.
//    If x(i)=0, then c(i)=0 and s(i) is chosen so that r(i) is real.
func Zlargv(n *int, x *mat.CVector, incx *int, y *mat.CVector, incy *int, c *mat.Vector, incc *int) {
	var czero, f, ff, fs, g, gs, r, sn complex128
	var cs, d, di, dr, eps, f2, f2s, g2, g2s, one, safmin, safmn2, safmx2, scale, two, zero float64
	var count, i, ic, ix, iy, j int

	two = 2.0
	one = 1.0
	zero = 0.0
	czero = (0.0 + 0.0*1i)

	safmin = Dlamch(SafeMinimum)
	eps = Dlamch(Epsilon)
	safmn2 = math.Pow(Dlamch(Base), float64(int(math.Log(safmin/eps)/math.Log(Dlamch(Base))/two)))
	safmx2 = one / safmn2

	ix = 1
	iy = 1
	ic = 1
	for i = 1; i <= (*n); i++ {
		f = x.Get(ix - 1)
		g = y.Get(iy - 1)

		//        Use identical algorithm as in ZLARTG
		scale = maxf64(abs1(f), abs1(g))
		fs = f
		gs = g
		count = 0
		if scale >= safmx2 {
		label10:
			;
			count = count + 1
			fs = fs * complex(safmn2, 0)
			gs = gs * complex(safmn2, 0)
			scale = scale * safmn2
			if scale >= safmx2 {
				goto label10
			}
		} else if scale <= safmn2 {
			if g == czero {
				cs = one
				sn = czero
				r = f
				goto label50
			}
		label20:
			;
			count = count - 1
			fs = fs * complex(safmx2, 0)
			gs = gs * complex(safmx2, 0)
			scale = scale * safmx2
			if scale <= safmn2 {
				goto label20
			}
		}
		f2 = abssq(fs)
		g2 = abssq(gs)
		if f2 <= maxf64(g2, one)*safmin {
			//           This is a rare case: F is very small.
			if f == czero {
				cs = zero
				r = complex(Dlapy2(toPtrf64(real(g)), toPtrf64(imag(g))), 0)
				//              Do complex/real division explicitly with two real
				//              divisions
				d = Dlapy2(toPtrf64(real(gs)), toPtrf64(imag(gs)))
				sn = complex(real(gs)/d, -imag(gs)/d)
				goto label50
			}
			f2s = Dlapy2(toPtrf64(real(fs)), toPtrf64(imag(fs)))
			//           G2 and G2S are accurate
			//           G2 is at least SAFMIN, and G2S is at least SAFMN2
			g2s = math.Sqrt(g2)
			//           Error in CS from underflow in F2S is at most
			//           UNFL / SAFMN2 .lt. sqrt(UNFL*EPS) .lt. EPS
			//           If max(G2,ONE)=G2, then F2 .lt. G2*SAFMIN,
			//           and so CS .lt. sqrt(SAFMIN)
			//           If max(G2,ONE)=ONE, then F2 .lt. SAFMIN
			//           and so CS .lt. sqrt(SAFMIN)/SAFMN2 = sqrt(EPS)
			//           Therefore, CS = F2S/G2S / sqrt( 1 + (F2S/G2S)**2 ) = F2S/G2S
			cs = f2s / g2s
			//           Make sure abs(FF) = 1
			//           Do complex/real division explicitly with 2 real divisions
			if abs1(f) > one {
				d = Dlapy2(toPtrf64(real(f)), toPtrf64(imag(f)))
				ff = complex(real(f)/d, imag(f)/d)
			} else {
				dr = safmx2 * real(f)
				di = safmx2 * imag(f)
				d = Dlapy2(&dr, &di)
				ff = complex(dr/d, di/d)
			}
			sn = ff * complex(real(gs)/g2s, -imag(gs)/g2s)
			r = complex(cs, 0)*f + sn*g
		} else {
			//           This is the most common case.
			//           Neither F2 nor F2/G2 are less than SAFMIN
			//           F2S cannot overflow, and it is accurate
			f2s = math.Sqrt(one + g2/f2)
			//           Do the F2S(real)*FS(complex) multiply with two real
			//           multiplies
			r = complex(f2s*real(fs), f2s*imag(fs))
			cs = one / f2s
			d = f2 + g2
			//           Do complex/real division explicitly with two real divisions
			sn = complex(real(r)/d, imag(r)/d)
			sn = sn * cmplx.Conj(gs)
			if count != 0 {
				if count > 0 {
					for j = 1; j <= count; j++ {
						r = r * complex(safmx2, 0)
					}
				} else {
					for j = 1; j <= -count; j++ {
						r = r * complex(safmn2, 0)
					}
				}
			}
		}
	label50:
		;
		c.Set(ic-1, cs)
		y.Set(iy-1, sn)
		x.Set(ix-1, r)
		ic = ic + (*incc)
		iy = iy + (*incy)
		ix = ix + (*incx)
	}
}
