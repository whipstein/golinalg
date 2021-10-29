package golapack

import (
	"math"
	"math/cmplx"
)

// Zlartg generates a plane rotation so that
//
//    [  CS  SN  ]     [ F ]     [ R ]
//    [  __      ]  .  [   ]  =  [   ]   where CS**2 + |SN|**2 = 1.
//    [ -SN  CS  ]     [ G ]     [ 0 ]
//
// This is a faster version of the BLAS1 routine ZROTG, except for
// the following differences:
//    F and G are unchanged on return.
//    If G=0, then CS=1 and SN=0.
//    If F=0, then CS=0 and SN is chosen so that R is real.
func Zlartg(f, g complex128) (cs float64, sn, r complex128) {
	var czero, ff, fs, gs complex128
	var d, di, dr, eps, f2, f2s, g2, g2s, one, safmin, safmn2, safmx2, scale, two, zero float64
	var count, i int

	two = 2.0
	one = 1.0
	zero = 0.0
	czero = (0.0 + 0.0*1i)

	safmin = Dlamch(SafeMinimum)
	eps = Dlamch(Epsilon)
	safmn2 = math.Pow(Dlamch(Base), float64(int((math.Log(safmin/eps)/math.Log(Dlamch(Base)))/two)))
	safmx2 = one / safmn2
	scale = math.Max(abs1(f), abs1(g))
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
		if g == czero || Disnan(int(cmplx.Abs(g))) {
			cs = one
			sn = czero
			r = f
			return
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
	if f2 <= math.Max(g2, one)*safmin {
		//        This is a rare case: F is very small.
		if f == czero {
			cs = zero
			r = complex(Dlapy2(real(g), imag(g)), 0)
			//           Do complex/real division explicitly with two real divisions
			d = Dlapy2(real(gs), imag(gs))
			sn = complex(real(gs)/d, -imag(gs)/d)
			return
		}
		f2s = Dlapy2(real(fs), imag(fs))
		//        G2 and G2S are accurate
		//        G2 is at least SAFMIN, and G2S is at least SAFMN2
		g2s = math.Sqrt(g2)
		//        Error in CS from underflow in F2S is at most
		//        UNFL / SAFMN2 .lt. math.Sqrt(UNFL*EPS) .lt. EPS
		//        If MAX(G2,ONE)=G2, then F2 .lt. G2*SAFMIN,
		//        and so CS .lt. math.Sqrt(SAFMIN)
		//        If MAX(G2,ONE)=ONE, then F2 .lt. SAFMIN
		//        and so CS .lt. math.Sqrt(SAFMIN)/SAFMN2 = math.Sqrt(EPS)
		//        Therefore, CS = F2S/G2S / math.Sqrt( 1 + (F2S/G2S)**2 ) = F2S/G2S
		cs = f2s / g2s
		//        Make sure abs(FF) = 1
		//        Do complex/real division explicitly with 2 real divisions
		if abs1(f) > one {
			d = Dlapy2(real(f), imag(f))
			ff = complex(real(f)/d, imag(f)/d)
		} else {
			dr = safmx2 * real(f)
			di = safmx2 * imag(f)
			d = Dlapy2(dr, di)
			ff = complex(dr/d, di/d)
		}
		sn = ff * complex(real(gs)/g2s, -imag(gs)/g2s)
		r = complex(cs, 0)*f + sn*g
	} else {
		//        This is the most common case.
		//        Neither F2 nor F2/G2 are less than SAFMIN
		//        F2S cannot overflow, and it is accurate
		f2s = math.Sqrt(one + g2/f2)
		//        Do the F2S(real)*FS(complex) multiply with two real multiplies
		r = complex(f2s*real(fs), f2s*imag(fs))
		cs = one / f2s
		d = f2 + g2
		//        Do complex/real division explicitly with two real divisions
		sn = complex(real(r)/d, imag(r)/d)
		sn = sn * cmplx.Conj(gs)
		if count != 0 {
			if count > 0 {
				for i = 1; i <= count; i++ {
					r = r * complex(safmx2, 0)
				}
			} else {
				for i = 1; i <= -count; i++ {
					r = r * complex(safmn2, 0)
				}
			}
		}
	}

	return
}
