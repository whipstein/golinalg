package golapack

import "math"

// Dlanv2 computes the Schur factorization of a real 2-by-2 nonsymmetric
// matrix in standard form:
//
//      [ A  B ] = [ CS -SN ] [ AA  BB ] [ CS  SN ]
//      [ C  D ]   [ SN  CS ] [ CC  DD ] [-SN  CS ]
//
// where either
// 1) CC = 0 so that AA and DD are real eigenvalues of the matrix, or
// 2) AA = DD and BB*CC < 0, so that AA + or - math.Sqrt(BB*CC) are complex
// conjugate eigenvalues.
func Dlanv2(a, b, c, d, rt1r, rt1i, rt2r, rt2i, cs, sn *float64) {
	var aa, bb, bcmax, bcmis, cc, cs1, dd, eps, half, multpl, one, p, sab, sac, scale, sigma, sn1, tau, temp, z, zero float64

	zero = 0.0
	half = 0.5
	one = 1.0
	multpl = 4.0

	eps = Dlamch(Precision)
	if (*c) == zero {
		(*cs) = one
		(*sn) = zero

	} else if (*b) == zero {
		//        Swap rows and columns
		(*cs) = zero
		(*sn) = one
		temp = (*d)
		(*d) = (*a)
		(*a) = temp
		(*b) = -(*c)
		(*c) = zero

	} else if ((*a)-(*d)) == zero && math.Copysign(one, *b) != math.Copysign(one, *c) {
		(*cs) = one
		(*sn) = zero

	} else {

		temp = (*a) - (*d)
		p = half * temp
		bcmax = math.Max(math.Abs(*b), math.Abs(*c))
		bcmis = math.Min(math.Abs(*b), math.Abs(*c)) * math.Copysign(one, *b) * math.Copysign(one, *c)
		scale = math.Max(math.Abs(p), bcmax)
		z = (p/scale)*p + (bcmax/scale)*bcmis

		//        If Z is of the order of the machine accuracy, postpone the
		//        decision on the nature of eigenvalues
		if z >= multpl*eps {
			//           Real eigenvalues. Compute A and D.
			z = p + math.Copysign(math.Sqrt(scale)*math.Sqrt(z), p)
			(*a) = (*d) + z
			(*d) = (*d) - (bcmax/z)*bcmis

			//           Compute B and the rotation matrix
			tau = Dlapy2(c, &z)
			(*cs) = z / tau
			(*sn) = (*c) / tau
			(*b) = (*b) - (*c)
			(*c) = zero

		} else {
			//           Complex eigenvalues, or real (almost) equal eigenvalues.
			//           Make diagonal elements equal.
			sigma = (*b) + (*c)
			tau = Dlapy2(&sigma, &temp)
			(*cs) = math.Sqrt(half * (one + math.Abs(sigma)/tau))
			(*sn) = -(p / (tau * (*cs))) * math.Copysign(one, sigma)

			//           Compute [ AA  BB ] = [ A  B ] [ CS -SN ]
			//                   [ CC  DD ]   [ C  D ] [ SN  CS ]
			aa = (*a)*(*cs) + (*b)*(*sn)
			bb = -(*a)*(*sn) + (*b)*(*cs)
			cc = (*c)*(*cs) + (*d)*(*sn)
			dd = -(*c)*(*sn) + (*d)*(*cs)

			//           Compute [ A  B ] = [ CS  SN ] [ AA  BB ]
			//                   [ C  D ]   [-SN  CS ] [ CC  DD ]
			(*a) = aa*(*cs) + cc*(*sn)
			(*b) = bb*(*cs) + dd*(*sn)
			(*c) = -aa*(*sn) + cc*(*cs)
			(*d) = -bb*(*sn) + dd*(*cs)

			temp = half * ((*a) + (*d))
			(*a) = temp
			(*d) = temp

			if (*c) != zero {
				if (*b) != zero {
					if math.Copysign(one, *b) == math.Copysign(one, *c) {
						//                    Real eigenvalues: reduce to upper triangular form
						sab = math.Sqrt(math.Abs(*b))
						sac = math.Sqrt(math.Abs(*c))
						p = math.Copysign(sab*sac, *c)
						tau = one / math.Sqrt(math.Abs((*b)+(*c)))
						(*a) = temp + p
						(*d) = temp - p
						(*b) = (*b) - (*c)
						(*c) = zero
						cs1 = sab * tau
						sn1 = sac * tau
						temp = (*cs)*cs1 - (*sn)*sn1
						(*sn) = (*cs)*sn1 + (*sn)*cs1
						(*cs) = temp
					}
				} else {
					(*b) = -(*c)
					(*c) = zero
					temp = (*cs)
					(*cs) = -(*sn)
					(*sn) = temp
				}
			}
		}

	}

	//     Store eigenvalues in (RT1R,RT1I) and (RT2R,RT2I).
	(*rt1r) = (*a)
	(*rt2r) = (*d)
	if (*c) == zero {
		(*rt1i) = zero
		(*rt2i) = zero
	} else {
		(*rt1i) = math.Sqrt(math.Abs(*b)) * math.Sqrt(math.Abs(*c))
		(*rt2i) = -(*rt1i)
	}
}
