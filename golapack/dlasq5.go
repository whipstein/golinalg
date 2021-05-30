package golapack

import "golinalg/mat"

// Dlasq5 computes one dqds transform in ping-pong form, one
// version for IEEE machines another for non IEEE machines.
func Dlasq5(i0, n0 *int, z *mat.Vector, pp *int, tau, sigma, dmin, dmin1, dmin2, dn, dnm1, dnm2 *float64, ieee *bool, eps *float64) {
	var d, dthresh, emin, half, temp, zero float64
	var j4, j4p2 int

	zero = 0.0
	half = 0.5
	if ((*n0) - (*i0) - 1) <= 0 {
		return
	}

	dthresh = (*eps) * ((*sigma) + (*tau))
	if (*tau) < dthresh*half {
		(*tau) = zero
	}
	if (*tau) != zero {
		j4 = 4*(*i0) + (*pp) - 3
		emin = z.Get(j4 + 4 - 1)
		d = z.Get(j4-1) - (*tau)
		(*dmin) = d
		(*dmin1) = -z.Get(j4 - 1)

		if *ieee {
			//        Code for IEEE arithmetic.
			if (*pp) == 0 {
				for j4 = 4 * (*i0); j4 <= 4*((*n0)-3); j4 += 4 {
					z.Set(j4-2-1, d+z.Get(j4-1-1))
					temp = z.Get(j4+1-1) / z.Get(j4-2-1)
					d = d*temp - (*tau)
					(*dmin) = minf64(*dmin, d)
					z.Set(j4-1, z.Get(j4-1-1)*temp)
					emin = minf64(z.Get(j4-1), emin)
				}
			} else {
				for j4 = 4 * (*i0); j4 <= 4*((*n0)-3); j4 += 4 {
					z.Set(j4-3-1, d+z.Get(j4-1))
					temp = z.Get(j4+2-1) / z.Get(j4-3-1)
					d = d*temp - (*tau)
					(*dmin) = minf64(*dmin, d)
					z.Set(j4-1-1, z.Get(j4-1)*temp)
					emin = minf64(z.Get(j4-1-1), emin)
				}
			}

			//        Unroll last two steps.
			(*dnm2) = d
			(*dmin2) = (*dmin)
			j4 = 4*((*n0)-2) - (*pp)
			j4p2 = j4 + 2*(*pp) - 1
			z.Set(j4-2-1, (*dnm2)+z.Get(j4p2-1))
			z.Set(j4-1, z.Get(j4p2+2-1)*(z.Get(j4p2-1)/z.Get(j4-2-1)))
			(*dnm1) = z.Get(j4p2+2-1)*((*dnm2)/z.Get(j4-2-1)) - (*tau)
			(*dmin) = minf64(*dmin, *dnm1)

			(*dmin1) = (*dmin)
			j4 = j4 + 4
			j4p2 = j4 + 2*(*pp) - 1
			z.Set(j4-2-1, (*dnm1)+z.Get(j4p2-1))
			z.Set(j4-1, z.Get(j4p2+2-1)*(z.Get(j4p2-1)/z.Get(j4-2-1)))
			(*dn) = z.Get(j4p2+2-1)*((*dnm1)/z.Get(j4-2-1)) - (*tau)
			(*dmin) = minf64(*dmin, *dn)

		} else {
			//        Code for non IEEE arithmetic.
			if (*pp) == 0 {
				for j4 = 4 * (*i0); j4 <= 4*((*n0)-3); j4 += 4 {
					z.Set(j4-2-1, d+z.Get(j4-1-1))
					if d < zero {
						return
					} else {
						z.Set(j4-1, z.Get(j4+1-1)*(z.Get(j4-1-1)/z.Get(j4-2-1)))
						d = z.Get(j4+1-1)*(d/z.Get(j4-2-1)) - (*tau)
					}
					(*dmin) = minf64(*dmin, d)
					emin = minf64(emin, z.Get(j4-1))
				}
			} else {
				for j4 = 4 * (*i0); j4 <= 4*((*n0)-3); j4 += 4 {
					z.Set(j4-3-1, d+z.Get(j4-1))
					if d < zero {
						return
					} else {
						z.Set(j4-1-1, z.Get(j4+2-1)*(z.Get(j4-1)/z.Get(j4-3-1)))
						d = z.Get(j4+2-1)*(d/z.Get(j4-3-1)) - (*tau)
					}
					(*dmin) = minf64(*dmin, d)
					emin = minf64(emin, z.Get(j4-1-1))
				}
			}

			//        Unroll last two steps.
			(*dnm2) = d
			(*dmin2) = (*dmin)
			j4 = 4*((*n0)-2) - (*pp)
			j4p2 = j4 + 2*(*pp) - 1
			z.Set(j4-2-1, (*dnm2)+z.Get(j4p2-1))
			if (*dnm2) < zero {
				return
			} else {
				z.Set(j4-1, z.Get(j4p2+2-1)*(z.Get(j4p2-1)/z.Get(j4-2-1)))
				(*dnm1) = z.Get(j4p2+2-1)*((*dnm2)/z.Get(j4-2-1)) - (*tau)
			}
			(*dmin) = minf64(*dmin, *dnm1)

			(*dmin1) = (*dmin)
			j4 = j4 + 4
			j4p2 = j4 + 2*(*pp) - 1
			z.Set(j4-2-1, (*dnm1)+z.Get(j4p2-1))
			if (*dnm1) < zero {
				return
			} else {
				z.Set(j4-1, z.Get(j4p2+2-1)*(z.Get(j4p2-1)/z.Get(j4-2-1)))
				(*dn) = z.Get(j4p2+2-1)*((*dnm1)/z.Get(j4-2-1)) - (*tau)
			}
			(*dmin) = minf64(*dmin, *dn)

		}
	} else {
		//     This is the version that sets d's to zero if they are small enough
		j4 = 4*(*i0) + (*pp) - 3
		emin = z.Get(j4 + 4 - 1)
		d = z.Get(j4-1) - (*tau)
		(*dmin) = d
		(*dmin1) = -z.Get(j4 - 1)
		if *ieee {
			//     Code for IEEE arithmetic.
			if (*pp) == 0 {
				for j4 = 4 * (*i0); j4 <= 4*((*n0)-3); j4 += 4 {
					z.Set(j4-2-1, d+z.Get(j4-1-1))
					temp = z.Get(j4+1-1) / z.Get(j4-2-1)
					d = d*temp - (*tau)
					if d < dthresh {
						d = zero
					}
					(*dmin) = minf64(*dmin, d)
					z.Set(j4-1, z.Get(j4-1-1)*temp)
					emin = minf64(z.Get(j4-1), emin)
				}
			} else {
				for j4 = 4 * (*i0); j4 <= 4*((*n0)-3); j4 += 4 {
					z.Set(j4-3-1, d+z.Get(j4-1))
					temp = z.Get(j4+2-1) / z.Get(j4-3-1)
					d = d*temp - (*tau)
					if d < dthresh {
						d = zero
					}
					(*dmin) = minf64(*dmin, d)
					z.Set(j4-1-1, z.Get(j4-1)*temp)
					emin = minf64(z.Get(j4-1-1), emin)
				}
			}

			//     Unroll last two steps.
			(*dnm2) = d
			(*dmin2) = (*dmin)
			j4 = 4*((*n0)-2) - (*pp)
			j4p2 = j4 + 2*(*pp) - 1
			z.Set(j4-2-1, (*dnm2)+z.Get(j4p2-1))
			z.Set(j4-1, z.Get(j4p2+2-1)*(z.Get(j4p2-1)/z.Get(j4-2-1)))
			(*dnm1) = z.Get(j4p2+2-1)*((*dnm2)/z.Get(j4-2-1)) - (*tau)
			(*dmin) = minf64(*dmin, *dnm1)
			//
			(*dmin1) = (*dmin)
			j4 = j4 + 4
			j4p2 = j4 + 2*(*pp) - 1
			z.Set(j4-2-1, (*dnm1)+z.Get(j4p2-1))
			z.Set(j4-1, z.Get(j4p2+2-1)*(z.Get(j4p2-1)/z.Get(j4-2-1)))
			(*dn) = z.Get(j4p2+2-1)*((*dnm1)/z.Get(j4-2-1)) - (*tau)
			(*dmin) = minf64(*dmin, *dn)

		} else {
			//     Code for non IEEE arithmetic.
			if (*pp) == 0 {
				for j4 = 4 * (*i0); j4 <= 4*((*n0)-3); j4 += 4 {
					z.Set(j4-2-1, d+z.Get(j4-1-1))
					if d < zero {
						return
					} else {
						z.Set(j4-1, z.Get(j4+1-1)*(z.Get(j4-1-1)/z.Get(j4-2-1)))
						d = z.Get(j4+1-1)*(d/z.Get(j4-2-1)) - (*tau)
					}
					if d < dthresh {
						d = zero
					}
					(*dmin) = minf64(*dmin, d)
					emin = minf64(emin, z.Get(j4-1))
				}
			} else {
				for j4 = 4 * (*i0); j4 <= 4*((*n0)-3); j4 += 4 {
					z.Set(j4-3-1, d+z.Get(j4-1))
					if d < zero {
						return
					} else {
						z.Set(j4-1-1, z.Get(j4+2-1)*(z.Get(j4-1)/z.Get(j4-3-1)))
						d = z.Get(j4+2-1)*(d/z.Get(j4-3-1)) - (*tau)
					}
					if d < dthresh {
						d = zero
					}
					(*dmin) = minf64(*dmin, d)
					emin = minf64(emin, z.Get(j4-1-1))
				}
			}

			//     Unroll last two steps.
			(*dnm2) = d
			(*dmin2) = (*dmin)
			j4 = 4*((*n0)-2) - (*pp)
			j4p2 = j4 + 2*(*pp) - 1
			z.Set(j4-2-1, (*dnm2)+z.Get(j4p2-1))
			if (*dnm2) < zero {
				return
			} else {
				z.Set(j4-1, z.Get(j4p2+2-1)*(z.Get(j4p2-1)/z.Get(j4-2-1)))
				(*dnm1) = z.Get(j4p2+2-1)*((*dnm2)/z.Get(j4-2-1)) - (*tau)
			}
			(*dmin) = minf64(*dmin, *dnm1)

			(*dmin1) = (*dmin)
			j4 = j4 + 4
			j4p2 = j4 + 2*(*pp) - 1
			z.Set(j4-2-1, (*dnm1)+z.Get(j4p2-1))
			if (*dnm1) < zero {
				return
			} else {
				z.Set(j4-1, z.Get(j4p2+2-1)*(z.Get(j4p2-1)/z.Get(j4-2-1)))
				(*dn) = z.Get(j4p2+2-1)*((*dnm1)/z.Get(j4-2-1)) - (*tau)
			}
			(*dmin) = minf64(*dmin, *dn)

		}
	}

	z.Set(j4+2-1, (*dn))
	z.Set(4*(*n0)-(*pp)-1, emin)
}
