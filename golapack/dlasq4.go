package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlasq4 computes an approximation TAU to the smallest eigenvalue
// using values of d from the previous transform.
func Dlasq4(i0, n0 int, z *mat.Vector, pp, n0in int, dmin, dmin1, dmin2, dn, dn1, dn2, g float64) (tau float64, ttype int, gOut float64) {
	var a2, b1, b2, cnst1, cnst2, cnst3, gam, gap1, gap2, half, hundrd, one, qurtr, s, third, two, zero float64
	var i4, nn, np int

	cnst1 = 0.5630
	cnst2 = 1.010
	cnst3 = 1.050
	qurtr = 0.250
	third = 0.3330
	half = 0.50
	zero = 0.0
	one = 1.0
	two = 2.0
	hundrd = 100.0
	gOut = g

	//     A negative DMIN forces the shift to take that absolute value
	//     TTYPE records the type of shift.
	if dmin <= zero {
		tau = -dmin
		ttype = -1
		return
	}

	nn = 4*n0 + pp
	if n0in == n0 {
		//        No eigenvalues deflated.
		if dmin == dn || dmin == dn1 {

			b1 = math.Sqrt(z.Get(nn-3-1)) * math.Sqrt(z.Get(nn-5-1))
			b2 = math.Sqrt(z.Get(nn-7-1)) * math.Sqrt(z.Get(nn-9-1))
			a2 = z.Get(nn-7-1) + z.Get(nn-5-1)

			//           Cases 2 and 3.
			if dmin == dn && dmin1 == dn1 {
				gap2 = dmin2 - a2 - dmin2*qurtr
				if gap2 > zero && gap2 > b2 {
					gap1 = a2 - dn - (b2/gap2)*b2
				} else {
					gap1 = a2 - dn - (b1 + b2)
				}
				if gap1 > zero && gap1 > b1 {
					s = math.Max(dn-(b1/gap1)*b1, half*dmin)
					ttype = -2
				} else {
					s = zero
					if dn > b1 {
						s = dn - b1
					}
					if a2 > (b1 + b2) {
						s = math.Min(s, a2-(b1+b2))
					}
					s = math.Max(s, third*dmin)
					ttype = -3
				}
			} else {
				//              Case 4.
				ttype = -4
				s = qurtr * dmin
				if dmin == dn {
					gam = dn
					a2 = zero
					if z.Get(nn-5-1) > z.Get(nn-7-1) {
						return
					}
					b2 = z.Get(nn-5-1) / z.Get(nn-7-1)
					np = nn - 9
				} else {
					np = nn - 2*pp
					gam = dn1
					if z.Get(np-4-1) > z.Get(np-2-1) {
						return
					}
					a2 = z.Get(np-4-1) / z.Get(np-2-1)
					if z.Get(nn-9-1) > z.Get(nn-11-1) {
						return
					}
					b2 = z.Get(nn-9-1) / z.Get(nn-11-1)
					np = nn - 13
				}

				//              Approximate contribution to norm squared from I < NN-1.
				a2 = a2 + b2
				for i4 = np; i4 >= 4*i0-1+pp; i4 -= 4 {
					if b2 == zero {
						goto label20
					}
					b1 = b2
					if z.Get(i4-1) > z.Get(i4-2-1) {
						return
					}
					b2 = b2 * (z.Get(i4-1) / z.Get(i4-2-1))
					a2 = a2 + b2
					if hundrd*math.Max(b2, b1) < a2 || cnst1 < a2 {
						goto label20
					}
				}
			label20:
				;
				a2 = cnst3 * a2

				//              Rayleigh quotient residual bound.
				if a2 < cnst1 {
					s = gam * (one - math.Sqrt(a2)) / (one + a2)
				}
			}
		} else if dmin == dn2 {
			//           Case 5.
			ttype = -5
			s = qurtr * dmin

			//           Compute contribution to norm squared from I > NN-2.
			np = nn - 2*pp
			b1 = z.Get(np - 2 - 1)
			b2 = z.Get(np - 6 - 1)
			gam = dn2
			if z.Get(np-8-1) > b2 || z.Get(np-4-1) > b1 {
				return
			}
			a2 = (z.Get(np-8-1) / b2) * (one + z.Get(np-4-1)/b1)

			//           Approximate contribution to norm squared from I < NN-2.
			if n0-i0 > 2 {
				b2 = z.Get(nn-13-1) / z.Get(nn-15-1)
				a2 = a2 + b2
				for i4 = nn - 17; i4 >= 4*i0-1+pp; i4 -= 4 {
					if b2 == zero {
						goto label40
					}
					b1 = b2
					if z.Get(i4-1) > z.Get(i4-2-1) {
						return
					}
					b2 = b2 * (z.Get(i4-1) / z.Get(i4-2-1))
					a2 = a2 + b2
					if hundrd*math.Max(b2, b1) < a2 || cnst1 < a2 {
						goto label40
					}
				}
			label40:
				;
				a2 = cnst3 * a2
			}

			if a2 < cnst1 {
				s = gam * (one - math.Sqrt(a2)) / (one + a2)
			}
		} else {
			//           Case 6, no information to guide us.
			if ttype == -6 {
				gOut = gOut + third*(one-gOut)
			} else if ttype == -18 {
				gOut = qurtr * third
			} else {
				gOut = qurtr
			}
			s = gOut * dmin
			ttype = -6
		}

	} else if n0in == (n0 + 1) {
		//        One eigenvalue just deflated. Use DMIN1, DN1 for DMIN and DN.
		if dmin1 == dn1 && dmin2 == dn2 {
			//           Cases 7 and 8.
			ttype = -7
			s = third * dmin1
			if z.Get(nn-5-1) > z.Get(nn-7-1) {
				return
			}
			b1 = z.Get(nn-5-1) / z.Get(nn-7-1)
			b2 = b1
			if b2 == zero {
				goto label60
			}
			for i4 = 4*n0 - 9 + pp; i4 >= 4*i0-1+pp; i4 -= 4 {
				a2 = b1
				if z.Get(i4-1) > z.Get(i4-2-1) {
					return
				}
				b1 = b1 * (z.Get(i4-1) / z.Get(i4-2-1))
				b2 = b2 + b1
				if hundrd*math.Max(b1, a2) < b2 {
					goto label60
				}
			}
		label60:
			;
			b2 = math.Sqrt(cnst3 * b2)
			a2 = dmin1 / (one + math.Pow(b2, 2))
			gap2 = half*dmin2 - a2
			if gap2 > zero && gap2 > b2*a2 {
				s = math.Max(s, a2*(one-cnst2*a2*(b2/gap2)*b2))
			} else {
				s = math.Max(s, a2*(one-cnst2*b2))
				ttype = -8
			}
		} else {
			//           Case 9.
			s = qurtr * dmin1
			if dmin1 == dn1 {
				s = half * dmin1
			}
			ttype = -9
		}

	} else if n0in == (n0 + 2) {
		//        Two eigenvalues deflated. Use DMIN2, DN2 for DMIN and DN.
		//
		//        Cases 10 and 11.
		if dmin2 == dn2 && two*z.Get(nn-5-1) < z.Get(nn-7-1) {
			ttype = -10
			s = third * dmin2
			if z.Get(nn-5-1) > z.Get(nn-7-1) {
				return
			}
			b1 = z.Get(nn-5-1) / z.Get(nn-7-1)
			b2 = b1
			if b2 == zero {
				goto label80
			}
			for i4 = 4*n0 - 9 + pp; i4 >= 4*i0-1+pp; i4 -= 4 {
				if z.Get(i4-1) > z.Get(i4-2-1) {
					return
				}
				b1 = b1 * (z.Get(i4-1) / z.Get(i4-2-1))
				b2 = b2 + b1
				if hundrd*b1 < b2 {
					goto label80
				}
			}
		label80:
			;
			b2 = math.Sqrt(cnst3 * b2)
			a2 = dmin2 / (one + math.Pow(b2, 2))
			gap2 = z.Get(nn-7-1) + z.Get(nn-9-1) - math.Sqrt(z.Get(nn-11-1))*math.Sqrt(z.Get(nn-9-1)) - a2
			if gap2 > zero && gap2 > b2*a2 {
				s = math.Max(s, a2*(one-cnst2*a2*(b2/gap2)*b2))
			} else {
				s = math.Max(s, a2*(one-cnst2*b2))
			}
		} else {
			s = qurtr * dmin2
			ttype = -11
		}
	} else if n0in > (n0 + 2) {
		//        Case 12, more than two eigenvalues deflated. No information.
		s = zero
		ttype = -12
	}

	tau = s

	return
}
