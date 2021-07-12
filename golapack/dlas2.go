package golapack

import "math"

// Dlas2 computes the singular values of the 2-by-2 matrix
//    [  F   G  ]
//    [  0   H  ].
// On return, SSMIN is the smaller singular value and SSMAX is the
// larger singular value.
func Dlas2(f, g, h, ssmin, ssmax *float64) {
	var as, at, au, c, fa, fhmn, fhmx, ga, ha, one, two, zero float64

	zero = 0.0
	one = 1.0
	two = 2.0

	fa = math.Abs(*f)
	ga = math.Abs(*g)
	ha = math.Abs(*h)
	fhmn = math.Min(fa, ha)
	fhmx = math.Max(fa, ha)
	if fhmn == zero {
		(*ssmin) = zero
		if fhmx == zero {
			(*ssmax) = ga
		} else {
			(*ssmax) = math.Max(fhmx, ga) * math.Sqrt(one+math.Pow(math.Min(fhmx, ga)/math.Max(fhmx, ga), 2))
		}
	} else {
		if ga < fhmx {
			as = one + fhmn/fhmx
			at = (fhmx - fhmn) / fhmx
			au = math.Pow(ga/fhmx, 2)
			c = two / (math.Sqrt(as*as+au) + math.Sqrt(at*at+au))
			(*ssmin) = fhmn * c
			(*ssmax) = fhmx / c
		} else {
			au = fhmx / ga
			if au == zero {
				//              Avoid possible harmful underflow if exponent range
				//              asymmetric (true SSMIN may not underflow even if
				//              AU underflows)
				(*ssmin) = (fhmn * fhmx) / ga
				(*ssmax) = ga
			} else {
				as = one + fhmn/fhmx
				at = (fhmx - fhmn) / fhmx
				c = one / (math.Sqrt(one+math.Pow(as*au, 2)) + math.Sqrt(one+math.Pow(at*au, 2)))
				(*ssmin) = (fhmn * c) * au
				(*ssmin) = (*ssmin) + (*ssmin)
				(*ssmax) = ga / (c + c)
			}
		}
	}
}
