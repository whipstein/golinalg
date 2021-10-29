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
func Dlanv2(a, b, c, d float64) (aOut, bOut, cOut, dOut, rt1r, rt1i, rt2r, rt2i, cs, sn float64) {
	var aa, bb, bcmax, bcmis, cc, cs1, dd, eps, half, multpl, one, p, sab, sac, scale, sigma, sn1, tau, temp, z, zero float64

	zero = 0.0
	half = 0.5
	one = 1.0
	multpl = 4.0
	aOut, bOut, cOut, dOut = a, b, c, d

	eps = Dlamch(Precision)
	if cOut == zero {
		cs = one
		sn = zero

	} else if bOut == zero {
		//        Swap rows and columns
		cs = zero
		sn = one
		temp = dOut
		dOut = aOut
		aOut = temp
		bOut = -cOut
		cOut = zero

	} else if (aOut-dOut) == zero && math.Copysign(one, bOut) != math.Copysign(one, cOut) {
		cs = one
		sn = zero

	} else {

		temp = aOut - dOut
		p = half * temp
		bcmax = math.Max(math.Abs(bOut), math.Abs(cOut))
		bcmis = math.Min(math.Abs(bOut), math.Abs(cOut)) * math.Copysign(one, bOut) * math.Copysign(one, cOut)
		scale = math.Max(math.Abs(p), bcmax)
		z = (p/scale)*p + (bcmax/scale)*bcmis

		//        If Z is of the order of the machine accuracy, postpone the
		//        decision on the nature of eigenvalues
		if z >= multpl*eps {
			//           Real eigenvalues. Compute A and D.
			z = p + math.Copysign(math.Sqrt(scale)*math.Sqrt(z), p)
			aOut = dOut + z
			dOut = dOut - (bcmax/z)*bcmis

			//           Compute B and the rotation matrix
			tau = Dlapy2(cOut, z)
			cs = z / tau
			sn = cOut / tau
			bOut = bOut - cOut
			cOut = zero

		} else {
			//           Complex eigenvalues, or real (almost) equal eigenvalues.
			//           Make diagonal elements equal.
			sigma = bOut + cOut
			tau = Dlapy2(sigma, temp)
			cs = math.Sqrt(half * (one + math.Abs(sigma)/tau))
			sn = -(p / (tau * cs)) * math.Copysign(one, sigma)

			//           Compute [ AA  BB ] = [ A  B ] [ CS -SN ]
			//                   [ CC  DD ]   [ C  D ] [ SN  CS ]
			aa = aOut*cs + bOut*sn
			bb = -aOut*sn + bOut*cs
			cc = cOut*cs + dOut*sn
			dd = -cOut*sn + dOut*cs

			//           Compute [ A  B ] = [ CS  SN ] [ AA  BB ]
			//                   [ C  D ]   [-SN  CS ] [ CC  DD ]
			aOut = aa*cs + cc*sn
			bOut = bb*cs + dd*sn
			cOut = -aa*sn + cc*cs
			dOut = -bb*sn + dd*cs

			temp = half * (aOut + dOut)
			aOut = temp
			dOut = temp

			if cOut != zero {
				if bOut != zero {
					if math.Copysign(one, bOut) == math.Copysign(one, cOut) {
						//                    Real eigenvalues: reduce to upper triangular form
						sab = math.Sqrt(math.Abs(bOut))
						sac = math.Sqrt(math.Abs(cOut))
						p = math.Copysign(sab*sac, cOut)
						tau = one / math.Sqrt(math.Abs(bOut+cOut))
						aOut = temp + p
						dOut = temp - p
						bOut = bOut - cOut
						cOut = zero
						cs1 = sab * tau
						sn1 = sac * tau
						temp = cs*cs1 - sn*sn1
						sn = cs*sn1 + sn*cs1
						cs = temp
					}
				} else {
					bOut = -cOut
					cOut = zero
					temp = cs
					cs = -sn
					sn = temp
				}
			}
		}

	}

	//     Store eigenvalues in (RT1R,RT1I) and (RT2R,RT2I).
	rt1r = aOut
	rt2r = dOut
	if cOut == zero {
		rt1i = zero
		rt2i = zero
	} else {
		rt1i = math.Sqrt(math.Abs(bOut)) * math.Sqrt(math.Abs(cOut))
		rt2i = -rt1i
	}

	return
}
