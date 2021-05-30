package golapack

import "math"

// Dlartgp generates a plane rotation so that
//
//    [  CS  SN  ]  .  [ F ]  =  [ R ]   where CS**2 + SN**2 = 1.
//    [ -SN  CS  ]     [ G ]     [ 0 ]
//
// This is a slower, more accurate version of the Level 1 BLAS routine DROTG,
// with the following other differences:
//    F and G are unchanged on return.
//    If G=0, then CS=(+/-)1 and SN=0.
//    If F=0 and (G .ne. 0), then CS=0 and SN=(+/-)1.
//
// The sign is chosen so that R >= 0.
func Dlartgp(f *float64, g *float64, cs *float64, sn *float64, r *float64) {
	var eps, f1, g1, one, safmin, safmn2, safmx2, scale, two, zero float64
	var count, i int

	zero = 0.0
	one = 1.0
	two = 2.0

	safmin = Dlamch(SafeMinimum)
	eps = Dlamch(Epsilon)
	safmn2 = math.Pow(Dlamch(Base), float64(int(math.Log(safmin/eps)/math.Log(Dlamch(Base))/two)))
	safmx2 = one / safmn2

	if (*g) == zero {
		(*cs) = signf64(one, *f)
		(*sn) = zero
		(*r) = math.Abs(*f)
	} else if (*f) == zero {
		(*cs) = zero
		(*sn) = signf64(one, *g)
		(*r) = math.Abs(*g)
	} else {
		f1 = (*f)
		g1 = (*g)
		scale = maxf64(math.Abs(f1), math.Abs(g1))
		if scale >= safmx2 {
			count = 0
		label10:
			;
			count = count + 1
			f1 = f1 * safmn2
			g1 = g1 * safmn2
			scale = maxf64(math.Abs(f1), math.Abs(g1))
			if scale >= safmx2 {
				goto label10
			}
			(*r) = math.Sqrt(math.Pow(f1, 2) + math.Pow(g1, 2))
			(*cs) = f1 / (*r)
			(*sn) = g1 / (*r)
			for i = 1; i <= count; i++ {
				(*r) = (*r) * safmx2
			}
		} else if scale <= safmn2 {
			count = 0
		label30:
			;
			count = count + 1
			f1 = f1 * safmx2
			g1 = g1 * safmx2
			scale = maxf64(math.Abs(f1), math.Abs(g1))
			if scale <= safmn2 {
				goto label30
			}
			(*r) = math.Sqrt(math.Pow(f1, 2) + math.Pow(g1, 2))
			(*cs) = f1 / (*r)
			(*sn) = g1 / (*r)
			for i = 1; i <= count; i++ {
				(*r) = (*r) * safmn2
			}
		} else {
			(*r) = math.Sqrt(math.Pow(f1, 2) + math.Pow(g1, 2))
			(*cs) = f1 / (*r)
			(*sn) = g1 / (*r)
		}
		if (*r) < zero {
			(*cs) = -(*cs)
			(*sn) = -(*sn)
			(*r) = -(*r)
		}
	}
}
