package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlasq6 computes one dqd (shift equal to zero) transform in
// ping-pong form, with protection against underflow and overflow.
func Dlasq6(i0, n0 *int, z *mat.Vector, pp *int, dmin, dmin1, dmin2, dn, dnm1, dnm2 *float64) {
	var d, emin, safmin, temp, zero float64
	var j4, j4p2 int

	zero = 0.0

	if ((*n0) - (*i0) - 1) <= 0 {
		return
	}

	safmin = Dlamch(SafeMinimum)
	j4 = 4*(*i0) + (*pp) - 3
	emin = z.Get(j4 + 4 - 1)
	d = z.Get(j4 - 1)
	(*dmin) = d

	if (*pp) == 0 {
		for j4 = 4 * (*i0); j4 <= 4*((*n0)-3); j4 += 4 {
			z.Set(j4-2-1, d+z.Get(j4-1-1))
			if z.Get(j4-2-1) == zero {
				z.Set(j4-1, zero)
				d = z.Get(j4 + 1 - 1)
				(*dmin) = d
				emin = zero
			} else if safmin*z.Get(j4) < z.Get(j4-2-1) && safmin*z.Get(j4-2-1) < z.Get(j4) {
				temp = z.Get(j4) / z.Get(j4-2-1)
				z.Set(j4-1, z.Get(j4-1-1)*temp)
				d = d * temp
			} else {
				z.Set(j4-1, z.Get(j4)*(z.Get(j4-1-1)/z.Get(j4-2-1)))
				d = z.Get(j4) * (d / z.Get(j4-2-1))
			}
			(*dmin) = math.Min(*dmin, d)
			emin = math.Min(emin, z.Get(j4-1))
		}
	} else {
		for j4 = 4 * (*i0); j4 <= 4*((*n0)-3); j4 += 4 {
			z.Set(j4-3-1, d+z.Get(j4-1))
			if z.Get(j4-3-1) == zero {
				z.Set(j4-1-1, zero)
				d = z.Get(j4 + 2 - 1)
				(*dmin) = d
				emin = zero
			} else if safmin*z.Get(j4+2-1) < z.Get(j4-3-1) && safmin*z.Get(j4-3-1) < z.Get(j4+2-1) {
				temp = z.Get(j4+2-1) / z.Get(j4-3-1)
				z.Set(j4-1-1, z.Get(j4-1)*temp)
				d = d * temp
			} else {
				z.Set(j4-1-1, z.Get(j4+2-1)*(z.Get(j4-1)/z.Get(j4-3-1)))
				d = z.Get(j4+2-1) * (d / z.Get(j4-3-1))
			}
			(*dmin) = math.Min(*dmin, d)
			emin = math.Min(emin, z.Get(j4-1-1))
		}
	}

	//     Unroll last two steps.
	(*dnm2) = d
	(*dmin2) = (*dmin)
	j4 = 4*((*n0)-2) - (*pp)
	j4p2 = j4 + 2*(*pp) - 1
	z.Set(j4-2-1, (*dnm2)+z.Get(j4p2-1))
	if z.Get(j4-2-1) == zero {
		z.Set(j4-1, zero)
		(*dnm1) = z.Get(j4p2 + 2 - 1)
		(*dmin) = (*dnm1)
		emin = zero
	} else if safmin*z.Get(j4p2+2-1) < z.Get(j4-2-1) && safmin*z.Get(j4-2-1) < z.Get(j4p2+2-1) {
		temp = z.Get(j4p2+2-1) / z.Get(j4-2-1)
		z.Set(j4-1, z.Get(j4p2-1)*temp)
		(*dnm1) = (*dnm2) * temp
	} else {
		z.Set(j4-1, z.Get(j4p2+2-1)*(z.Get(j4p2-1)/z.Get(j4-2-1)))
		(*dnm1) = z.Get(j4p2+2-1) * ((*dnm2) / z.Get(j4-2-1))
	}
	(*dmin) = math.Min(*dmin, *dnm1)

	(*dmin1) = (*dmin)
	j4 = j4 + 4
	j4p2 = j4 + 2*(*pp) - 1
	z.Set(j4-2-1, (*dnm1)+z.Get(j4p2-1))
	if z.Get(j4-2-1) == zero {
		z.Set(j4-1, zero)
		(*dn) = z.Get(j4p2 + 2 - 1)
		(*dmin) = (*dn)
		emin = zero
	} else if safmin*z.Get(j4p2+2-1) < z.Get(j4-2-1) && safmin*z.Get(j4-2-1) < z.Get(j4p2+2-1) {
		temp = z.Get(j4p2+2-1) / z.Get(j4-2-1)
		z.Set(j4-1, z.Get(j4p2-1)*temp)
		(*dn) = (*dnm1) * temp
	} else {
		z.Set(j4-1, z.Get(j4p2+2-1)*(z.Get(j4p2-1)/z.Get(j4-2-1)))
		(*dn) = z.Get(j4p2+2-1) * ((*dnm1) / z.Get(j4-2-1))
	}
	(*dmin) = math.Min(*dmin, *dn)

	z.Set(j4+2-1, (*dn))
	z.Set(4*(*n0)-(*pp)-1, emin)
}
