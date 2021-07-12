package golapack

import "math"

// Dladiv performs complex division in  real arithmetic
//
//                       a + i*b
//            p + i*q = ---------
//                       c + i*d
//
// The algorithm is due to Michael Baudin and Robert L. Smith
// and can be found in the paper
// "A Robust Complex Division in Scilab"
func Dladiv(a, b, c, d, p, q *float64) {
	var aa, ab, bb, be, bs, cc, cd, dd, eps, half, ov, s, two, un float64

	bs = 2.0
	half = 0.5
	two = 2.0

	aa = (*a)
	bb = (*b)
	cc = (*c)
	dd = (*d)
	ab = math.Max(math.Abs(*a), math.Abs(*b))
	cd = math.Max(math.Abs(*c), math.Abs(*d))
	s = 1.0
	ov = Dlamch(Overflow)
	un = Dlamch(SafeMinimum)
	eps = Dlamch(Epsilon)
	be = bs / (eps * eps)
	if ab >= half*ov {
		aa = half * aa
		bb = half * bb
		s = two * s
	}
	if cd >= half*ov {
		cc = half * cc
		dd = half * dd
		s = half * s
	}
	if ab <= un*bs/eps {
		aa = aa * be
		bb = bb * be
		s = s / be
	}
	if cd <= un*bs/eps {
		cc = cc * be
		dd = dd * be
		s = s * be
	}
	if math.Abs(*d) <= math.Abs(*c) {
		Dladiv1(&aa, &bb, &cc, &dd, p, q)
	} else {
		Dladiv1(&bb, &aa, &dd, &cc, p, q)
		(*q) = -(*q)
	}
	(*p) = (*p) * s
	(*q) = (*q) * s
}

// \ingroup doubleOTHERauxiliary
func Dladiv1(a, b, c, d, p, q *float64) {
	var one, r, t float64

	one = 1.0

	r = (*d) / (*c)
	t = one / ((*c) + (*d)*r)
	(*p) = Dladiv2(a, b, c, d, &r, &t)
	(*a) = -(*a)
	(*q) = Dladiv2(b, a, c, d, &r, &t)
}

// \ingroup doubleOTHERauxiliary
func Dladiv2(a, b, c, d, r, t *float64) (dladiv2Return float64) {
	var br, zero float64

	zero = 0.0

	if (*r) != zero {
		br = (*b) * (*r)
		if br != zero {
			dladiv2Return = ((*a) + br) * (*t)
		} else {
			dladiv2Return = (*a)*(*t) + ((*b)*(*t))*(*r)
		}
	} else {
		dladiv2Return = ((*a) + (*d)*((*b)/(*c))) * (*t)
	}

	return
}
