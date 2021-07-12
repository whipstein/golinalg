package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlaed4 This subroutine computes the I-th updated eigenvalue of a symmetric
// rank-one modification to a diagonal matrix whose elements are
// given in the array d, and that
//
//            D(i) < D(j)  for  i < j
//
// and that RHO > 0.  This is arranged by the calling routine, and is
// no loss in generality.  The rank-one modified system is thus
//
//            diag( D )  +  RHO * Z * Z_transpose.
//
// where we assume the Euclidean norm of Z is 1.
//
// The method consists of approximating the rational functions in the
// secular equation by simpler interpolating rational functions.
func Dlaed4(n, i *int, d, z, delta *mat.Vector, rho, dlam *float64, info *int) {
	var orgati, swtch, swtch3 bool
	var a, b, c, del, dltlb, dltub, dphi, dpsi, dw, eight, eps, erretm, eta, four, midpt, one, phi, prew, psi, rhoinv, tau, temp, temp1, ten, three, two, w, zero float64
	var ii, iim1, iip1, ip1, iter, j, maxit, niter int

	zz := vf(3)

	maxit = 30
	zero = 0.0
	one = 1.0
	two = 2.0
	three = 3.0
	four = 4.0
	eight = 8.0
	ten = 10.0

	//     Since this routine is called in an inner loop, we do no argument
	//     checking.
	//
	//     Quick return for N=1 and 2.
	(*info) = 0
	if (*n) == 1 {
		//         Presumably, I=1 upon entry
		(*dlam) = d.Get(0) + (*rho)*z.Get(0)*z.Get(0)
		delta.Set(0, one)
		return
	}
	if (*n) == 2 {
		Dlaed5(i, d, z, delta, rho, dlam)
		return
	}

	//     Compute machine epsilon
	eps = Dlamch(Epsilon)
	rhoinv = one / (*rho)

	//     The case I = N
	if (*i) == (*n) {
		//        Initialize some basic variables
		ii = (*n) - 1
		niter = 1

		//        Calculate initial guess
		midpt = (*rho) / two

		//        If ||Z||_2 is not one, then TEMP should be set to
		//        RHO * ||Z||_2^2 / TWO
		for j = 1; j <= (*n); j++ {
			delta.Set(j-1, (d.Get(j-1)-d.Get((*i)-1))-midpt)
		}

		psi = zero
		for j = 1; j <= (*n)-2; j++ {
			psi = psi + z.Get(j-1)*z.Get(j-1)/delta.Get(j-1)
		}

		c = rhoinv + psi
		w = c + z.Get(ii-1)*z.Get(ii-1)/delta.Get(ii-1) + z.Get((*n)-1)*z.Get((*n)-1)/delta.Get((*n)-1)

		if w <= zero {
			temp = z.Get((*n)-1-1)*z.Get((*n)-1-1)/(d.Get((*n)-1)-d.Get((*n)-1-1)+(*rho)) + z.Get((*n)-1)*z.Get((*n)-1)/(*rho)
			if c <= temp {
				tau = (*rho)
			} else {
				del = d.Get((*n)-1) - d.Get((*n)-1-1)
				a = -c*del + z.Get((*n)-1-1)*z.Get((*n)-1-1) + z.Get((*n)-1)*z.Get((*n)-1)
				b = z.Get((*n)-1) * z.Get((*n)-1) * del
				if a < zero {
					tau = two * b / (math.Sqrt(a*a+four*b*c) - a)
				} else {
					tau = (a + math.Sqrt(a*a+four*b*c)) / (two * c)
				}
			}

			//           It can be proved that
			//               D(N)+RHO/2 <= LAMBDA(N) < D(N)+TAU <= D(N)+RHO
			dltlb = midpt
			dltub = (*rho)
		} else {
			del = d.Get((*n)-1) - d.Get((*n)-1-1)
			a = -c*del + z.Get((*n)-1-1)*z.Get((*n)-1-1) + z.Get((*n)-1)*z.Get((*n)-1)
			b = z.Get((*n)-1) * z.Get((*n)-1) * del
			if a < zero {
				tau = two * b / (math.Sqrt(a*a+four*b*c) - a)
			} else {
				tau = (a + math.Sqrt(a*a+four*b*c)) / (two * c)
			}

			//           It can be proved that
			//               D(N) < D(N)+TAU < LAMBDA(N) < D(N)+RHO/2
			dltlb = zero
			dltub = midpt
		}

		for j = 1; j <= (*n); j++ {
			delta.Set(j-1, (d.Get(j-1)-d.Get((*i)-1))-tau)
		}

		//        Evaluate PSI and the derivative DPSI
		dpsi = zero
		psi = zero
		erretm = zero
		for j = 1; j <= ii; j++ {
			temp = z.Get(j-1) / delta.Get(j-1)
			psi = psi + z.Get(j-1)*temp
			dpsi = dpsi + temp*temp
			erretm = erretm + psi
		}
		erretm = math.Abs(erretm)

		//        Evaluate PHI and the derivative DPHI
		temp = z.Get((*n)-1) / delta.Get((*n)-1)
		phi = z.Get((*n)-1) * temp
		dphi = temp * temp
		erretm = eight*(-phi-psi) + erretm - phi + rhoinv + math.Abs(tau)*(dpsi+dphi)

		w = rhoinv + phi + psi

		//        Test for convergence
		if math.Abs(w) <= eps*erretm {
			(*dlam) = d.Get((*i)-1) + tau
			return
		}

		if w <= zero {
			dltlb = math.Max(dltlb, tau)
		} else {
			dltub = math.Min(dltub, tau)
		}

		//        Calculate the new step
		niter = niter + 1
		c = w - delta.Get((*n)-1-1)*dpsi - delta.Get((*n)-1)*dphi
		a = (delta.Get((*n)-1-1)+delta.Get((*n)-1))*w - delta.Get((*n)-1-1)*delta.Get((*n)-1)*(dpsi+dphi)
		b = delta.Get((*n)-1-1) * delta.Get((*n)-1) * w
		if c < zero {
			c = math.Abs(c)
		}
		if c == zero {
			//          ETA = B/A
			//           ETA = RHO - TAU
			eta = dltub - tau
		} else if a >= zero {
			eta = (a + math.Sqrt(math.Abs(a*a-four*b*c))) / (two * c)
		} else {
			eta = two * b / (a - math.Sqrt(math.Abs(a*a-four*b*c)))
		}

		//        Note, eta should be positive if w is negative, and
		//        eta should be negative otherwise. However,
		//        if for some reason caused by roundoff, eta*w > 0,
		//        we simply use one Newton step instead. This way
		//        will guarantee eta*w < 0.
		if w*eta > zero {
			eta = -w / (dpsi + dphi)
		}
		temp = tau + eta
		if temp > dltub || temp < dltlb {
			if w < zero {
				eta = (dltub - tau) / two
			} else {
				eta = (dltlb - tau) / two
			}
		}
		for j = 1; j <= (*n); j++ {
			delta.Set(j-1, delta.Get(j-1)-eta)
		}

		tau = tau + eta

		//        Evaluate PSI and the derivative DPSI
		dpsi = zero
		psi = zero
		erretm = zero
		for j = 1; j <= ii; j++ {
			temp = z.Get(j-1) / delta.Get(j-1)
			psi = psi + z.Get(j-1)*temp
			dpsi = dpsi + temp*temp
			erretm = erretm + psi
		}
		erretm = math.Abs(erretm)

		//        Evaluate PHI and the derivative DPHI
		temp = z.Get((*n)-1) / delta.Get((*n)-1)
		phi = z.Get((*n)-1) * temp
		dphi = temp * temp
		erretm = eight*(-phi-psi) + erretm - phi + rhoinv + math.Abs(tau)*(dpsi+dphi)

		w = rhoinv + phi + psi

		//        Main loop to update the values of the array   DELTA
		iter = niter + 1

		for niter = iter; niter <= maxit; niter++ {
			//           Test for convergence
			if math.Abs(w) <= eps*erretm {
				(*dlam) = d.Get((*i)-1) + tau
				return
			}

			if w <= zero {
				dltlb = math.Max(dltlb, tau)
			} else {
				dltub = math.Min(dltub, tau)
			}

			//           Calculate the new step
			c = w - delta.Get((*n)-1-1)*dpsi - delta.Get((*n)-1)*dphi
			a = (delta.Get((*n)-1-1)+delta.Get((*n)-1))*w - delta.Get((*n)-1-1)*delta.Get((*n)-1)*(dpsi+dphi)
			b = delta.Get((*n)-1-1) * delta.Get((*n)-1) * w
			if a >= zero {
				eta = (a + math.Sqrt(math.Abs(a*a-four*b*c))) / (two * c)
			} else {
				eta = two * b / (a - math.Sqrt(math.Abs(a*a-four*b*c)))
			}

			//           Note, eta should be positive if w is negative, and
			//           eta should be negative otherwise. However,
			//           if for some reason caused by roundoff, eta*w > 0,
			//           we simply use one Newton step instead. This way
			//           will guarantee eta*w < 0.
			if w*eta > zero {
				eta = -w / (dpsi + dphi)
			}
			temp = tau + eta
			if temp > dltub || temp < dltlb {
				if w < zero {
					eta = (dltub - tau) / two
				} else {
					eta = (dltlb - tau) / two
				}
			}
			for j = 1; j <= (*n); j++ {
				delta.Set(j-1, delta.Get(j-1)-eta)
			}

			tau = tau + eta

			//           Evaluate PSI and the derivative DPSI
			dpsi = zero
			psi = zero
			erretm = zero
			for j = 1; j <= ii; j++ {
				temp = z.Get(j-1) / delta.Get(j-1)
				psi = psi + z.Get(j-1)*temp
				dpsi = dpsi + temp*temp
				erretm = erretm + psi
			}
			erretm = math.Abs(erretm)

			//           Evaluate PHI and the derivative DPHI
			temp = z.Get((*n)-1) / delta.Get((*n)-1)
			phi = z.Get((*n)-1) * temp
			dphi = temp * temp
			erretm = eight*(-phi-psi) + erretm - phi + rhoinv + math.Abs(tau)*(dpsi+dphi)

			w = rhoinv + phi + psi
		}

		//        Return with INFO = 1, NITER = MAXIT and not converged
		(*info) = 1
		(*dlam) = d.Get((*i)-1) + tau
		return

		//        End for the case I = N
	} else {
		//        The case for I < N
		niter = 1
		ip1 = (*i) + 1

		//        Calculate initial guess
		del = d.Get(ip1-1) - d.Get((*i)-1)
		midpt = del / two
		for j = 1; j <= (*n); j++ {
			delta.Set(j-1, (d.Get(j-1)-d.Get((*i)-1))-midpt)
		}

		psi = zero
		for j = 1; j <= (*i)-1; j++ {
			psi = psi + z.Get(j-1)*z.Get(j-1)/delta.Get(j-1)
		}

		phi = zero
		for j = (*n); j >= (*i)+2; j-- {
			phi = phi + z.Get(j-1)*z.Get(j-1)/delta.Get(j-1)
		}
		c = rhoinv + psi + phi
		w = c + z.Get((*i)-1)*z.Get((*i)-1)/delta.Get((*i)-1) + z.Get(ip1-1)*z.Get(ip1-1)/delta.Get(ip1-1)

		if w > zero {
			//           d(i)< the ith eigenvalue < (d(i)+d(i+1))/2
			//
			//           We choose d(i) as origin.
			orgati = true
			a = c*del + z.Get((*i)-1)*z.Get((*i)-1) + z.Get(ip1-1)*z.Get(ip1-1)
			b = z.Get((*i)-1) * z.Get((*i)-1) * del
			if a > zero {
				tau = two * b / (a + math.Sqrt(math.Abs(a*a-four*b*c)))
			} else {
				tau = (a - math.Sqrt(math.Abs(a*a-four*b*c))) / (two * c)
			}
			dltlb = zero
			dltub = midpt
		} else {
			//           (d(i)+d(i+1))/2 <= the ith eigenvalue < d(i+1)
			//
			//           We choose d(i+1) as origin.
			orgati = false
			a = c*del - z.Get((*i)-1)*z.Get((*i)-1) - z.Get(ip1-1)*z.Get(ip1-1)
			b = z.Get(ip1-1) * z.Get(ip1-1) * del
			if a < zero {
				tau = two * b / (a - math.Sqrt(math.Abs(a*a+four*b*c)))
			} else {
				tau = -(a + math.Sqrt(math.Abs(a*a+four*b*c))) / (two * c)
			}
			dltlb = -midpt
			dltub = zero
		}

		if orgati {
			for j = 1; j <= (*n); j++ {
				delta.Set(j-1, (d.Get(j-1)-d.Get((*i)-1))-tau)
			}
		} else {
			for j = 1; j <= (*n); j++ {
				delta.Set(j-1, (d.Get(j-1)-d.Get(ip1-1))-tau)
			}
		}
		if orgati {
			ii = (*i)
		} else {
			ii = (*i) + 1
		}
		iim1 = ii - 1
		iip1 = ii + 1

		//        Evaluate PSI and the derivative DPSI
		dpsi = zero
		psi = zero
		erretm = zero
		for j = 1; j <= iim1; j++ {
			temp = z.Get(j-1) / delta.Get(j-1)
			psi = psi + z.Get(j-1)*temp
			dpsi = dpsi + temp*temp
			erretm = erretm + psi
		}
		erretm = math.Abs(erretm)

		//        Evaluate PHI and the derivative DPHI
		dphi = zero
		phi = zero
		for j = (*n); j >= iip1; j-- {
			temp = z.Get(j-1) / delta.Get(j-1)
			phi = phi + z.Get(j-1)*temp
			dphi = dphi + temp*temp
			erretm = erretm + phi
		}

		w = rhoinv + phi + psi

		//        W is the value of the secular function with
		//        its ii-th element removed.
		swtch3 = false
		if orgati {
			if w < zero {
				swtch3 = true
			}
		} else {
			if w > zero {
				swtch3 = true
			}
		}
		if ii == 1 || ii == (*n) {
			swtch3 = false
		}

		temp = z.Get(ii-1) / delta.Get(ii-1)
		dw = dpsi + dphi + temp*temp
		temp = z.Get(ii-1) * temp
		w = w + temp
		erretm = eight*(phi-psi) + erretm + two*rhoinv + three*math.Abs(temp) + math.Abs(tau)*dw

		//        Test for convergence
		if math.Abs(w) <= eps*erretm {
			if orgati {
				(*dlam) = d.Get((*i)-1) + tau
			} else {
				(*dlam) = d.Get(ip1-1) + tau
			}
			return
		}
		//
		if w <= zero {
			dltlb = math.Max(dltlb, tau)
		} else {
			dltub = math.Min(dltub, tau)
		}

		//        Calculate the new step
		niter = niter + 1
		if !swtch3 {
			if orgati {
				c = w - delta.Get(ip1-1)*dw - (d.Get((*i)-1)-d.Get(ip1-1))*math.Pow(z.Get((*i)-1)/delta.Get((*i)-1), 2)
			} else {
				c = w - delta.Get((*i)-1)*dw - (d.Get(ip1-1)-d.Get((*i)-1))*math.Pow(z.Get(ip1-1)/delta.Get(ip1-1), 2)
			}
			a = (delta.Get((*i)-1)+delta.Get(ip1-1))*w - delta.Get((*i)-1)*delta.Get(ip1-1)*dw
			b = delta.Get((*i)-1) * delta.Get(ip1-1) * w
			if c == zero {
				if a == zero {
					if orgati {
						a = z.Get((*i)-1)*z.Get((*i)-1) + delta.Get(ip1-1)*delta.Get(ip1-1)*(dpsi+dphi)
					} else {
						a = z.Get(ip1-1)*z.Get(ip1-1) + delta.Get((*i)-1)*delta.Get((*i)-1)*(dpsi+dphi)
					}
				}
				eta = b / a
			} else if a <= zero {
				eta = (a - math.Sqrt(math.Abs(a*a-four*b*c))) / (two * c)
			} else {
				eta = two * b / (a + math.Sqrt(math.Abs(a*a-four*b*c)))
			}
		} else {
			//           Interpolation using THREE most relevant poles
			temp = rhoinv + psi + phi
			if orgati {
				temp1 = z.Get(iim1-1) / delta.Get(iim1-1)
				temp1 = temp1 * temp1
				c = temp - delta.Get(iip1-1)*(dpsi+dphi) - (d.Get(iim1-1)-d.Get(iip1-1))*temp1
				zz.Set(0, z.Get(iim1-1)*z.Get(iim1-1))
				zz.Set(2, delta.Get(iip1-1)*delta.Get(iip1-1)*((dpsi-temp1)+dphi))
			} else {
				temp1 = z.Get(iip1-1) / delta.Get(iip1-1)
				temp1 = temp1 * temp1
				c = temp - delta.Get(iim1-1)*(dpsi+dphi) - (d.Get(iip1-1)-d.Get(iim1-1))*temp1
				zz.Set(0, delta.Get(iim1-1)*delta.Get(iim1-1)*(dpsi+(dphi-temp1)))
				zz.Set(2, z.Get(iip1-1)*z.Get(iip1-1))
			}
			zz.Set(1, z.Get(ii-1)*z.Get(ii-1))
			Dlaed6(&niter, orgati, &c, delta.Off(iim1-1), zz, &w, &eta, info)
			if (*info) != 0 {
				return
			}
		}

		//        Note, eta should be positive if w is negative, and
		//        eta should be negative otherwise. However,
		//        if for some reason caused by roundoff, eta*w > 0,
		//        we simply use one Newton step instead. This way
		//        will guarantee eta*w < 0.
		if w*eta >= zero {
			eta = -w / dw
		}
		temp = tau + eta
		if temp > dltub || temp < dltlb {
			if w < zero {
				eta = (dltub - tau) / two
			} else {
				eta = (dltlb - tau) / two
			}
		}

		prew = w

		for j = 1; j <= (*n); j++ {
			delta.Set(j-1, delta.Get(j-1)-eta)
		}

		//        Evaluate PSI and the derivative DPSI
		dpsi = zero
		psi = zero
		erretm = zero
		for j = 1; j <= iim1; j++ {
			temp = z.Get(j-1) / delta.Get(j-1)
			psi = psi + z.Get(j-1)*temp
			dpsi = dpsi + temp*temp
			erretm = erretm + psi
		}
		erretm = math.Abs(erretm)

		//        Evaluate PHI and the derivative DPHI
		dphi = zero
		phi = zero
		for j = (*n); j >= iip1; j-- {
			temp = z.Get(j-1) / delta.Get(j-1)
			phi = phi + z.Get(j-1)*temp
			dphi = dphi + temp*temp
			erretm = erretm + phi
		}

		temp = z.Get(ii-1) / delta.Get(ii-1)
		dw = dpsi + dphi + temp*temp
		temp = z.Get(ii-1) * temp
		w = rhoinv + phi + psi + temp
		erretm = eight*(phi-psi) + erretm + two*rhoinv + three*math.Abs(temp) + math.Abs(tau+eta)*dw

		swtch = false
		if orgati {
			if -w > math.Abs(prew)/ten {
				swtch = true
			}
		} else {
			if w > math.Abs(prew)/ten {
				swtch = true
			}
		}

		tau = tau + eta

		//        Main loop to update the values of the array   DELTA
		iter = niter + 1

		for niter = iter; niter <= maxit; niter++ {
			//           Test for convergence
			if math.Abs(w) <= eps*erretm {
				if orgati {
					(*dlam) = d.Get((*i)-1) + tau
				} else {
					(*dlam) = d.Get(ip1-1) + tau
				}
				return
			}

			if w <= zero {
				dltlb = math.Max(dltlb, tau)
			} else {
				dltub = math.Min(dltub, tau)
			}

			//           Calculate the new step
			if !swtch3 {
				if !swtch {
					if orgati {
						c = w - delta.Get(ip1-1)*dw - (d.Get((*i)-1)-d.Get(ip1-1))*math.Pow(z.Get((*i)-1)/delta.Get((*i)-1), 2)
					} else {
						c = w - delta.Get((*i)-1)*dw - (d.Get(ip1-1)-d.Get((*i)-1))*math.Pow(z.Get(ip1-1)/delta.Get(ip1-1), 2)
					}
				} else {
					temp = z.Get(ii-1) / delta.Get(ii-1)
					if orgati {
						dpsi = dpsi + temp*temp
					} else {
						dphi = dphi + temp*temp
					}
					c = w - delta.Get((*i)-1)*dpsi - delta.Get(ip1-1)*dphi
				}
				a = (delta.Get((*i)-1)+delta.Get(ip1-1))*w - delta.Get((*i)-1)*delta.Get(ip1-1)*dw
				b = delta.Get((*i)-1) * delta.Get(ip1-1) * w
				if c == zero {
					if a == zero {
						if !swtch {
							if orgati {
								a = z.Get((*i)-1)*z.Get((*i)-1) + delta.Get(ip1-1)*delta.Get(ip1-1)*(dpsi+dphi)
							} else {
								a = z.Get(ip1-1)*z.Get(ip1-1) + delta.Get((*i)-1)*delta.Get((*i)-1)*(dpsi+dphi)
							}
						} else {
							a = delta.Get((*i)-1)*delta.Get((*i)-1)*dpsi + delta.Get(ip1-1)*delta.Get(ip1-1)*dphi
						}
					}
					eta = b / a
				} else if a <= zero {
					eta = (a - math.Sqrt(math.Abs(a*a-four*b*c))) / (two * c)
				} else {
					eta = two * b / (a + math.Sqrt(math.Abs(a*a-four*b*c)))
				}
			} else {
				//              Interpolation using THREE most relevant poles
				temp = rhoinv + psi + phi
				if swtch {
					c = temp - delta.Get(iim1-1)*dpsi - delta.Get(iip1-1)*dphi
					zz.Set(0, delta.Get(iim1-1)*delta.Get(iim1-1)*dpsi)
					zz.Set(2, delta.Get(iip1-1)*delta.Get(iip1-1)*dphi)
				} else {
					if orgati {
						temp1 = z.Get(iim1-1) / delta.Get(iim1-1)
						temp1 = temp1 * temp1
						c = temp - delta.Get(iip1-1)*(dpsi+dphi) - (d.Get(iim1-1)-d.Get(iip1-1))*temp1
						zz.Set(0, z.Get(iim1-1)*z.Get(iim1-1))
						zz.Set(2, delta.Get(iip1-1)*delta.Get(iip1-1)*((dpsi-temp1)+dphi))
					} else {
						temp1 = z.Get(iip1-1) / delta.Get(iip1-1)
						temp1 = temp1 * temp1
						c = temp - delta.Get(iim1-1)*(dpsi+dphi) - (d.Get(iip1-1)-d.Get(iim1-1))*temp1
						zz.Set(0, delta.Get(iim1-1)*delta.Get(iim1-1)*(dpsi+(dphi-temp1)))
						zz.Set(2, z.Get(iip1-1)*z.Get(iip1-1))
					}
				}
				Dlaed6(&niter, orgati, &c, delta.Off(iim1-1), zz, &w, &eta, info)
				if (*info) != 0 {
					return
				}
			}

			//           Note, eta should be positive if w is negative, and
			//           eta should be negative otherwise. However,
			//           if for some reason caused by roundoff, eta*w > 0,
			//           we simply use one Newton step instead. This way
			//           will guarantee eta*w < 0.
			if w*eta >= zero {
				eta = -w / dw
			}
			temp = tau + eta
			if temp > dltub || temp < dltlb {
				if w < zero {
					eta = (dltub - tau) / two
				} else {
					eta = (dltlb - tau) / two
				}
			}

			for j = 1; j <= (*n); j++ {
				delta.Set(j-1, delta.Get(j-1)-eta)
			}

			tau = tau + eta
			prew = w

			//           Evaluate PSI and the derivative DPSI
			dpsi = zero
			psi = zero
			erretm = zero
			for j = 1; j <= iim1; j++ {
				temp = z.Get(j-1) / delta.Get(j-1)
				psi = psi + z.Get(j-1)*temp
				dpsi = dpsi + temp*temp
				erretm = erretm + psi
			}
			erretm = math.Abs(erretm)

			//           Evaluate PHI and the derivative DPHI
			dphi = zero
			phi = zero
			for j = (*n); j >= iip1; j-- {
				temp = z.Get(j-1) / delta.Get(j-1)
				phi = phi + z.Get(j-1)*temp
				dphi = dphi + temp*temp
				erretm = erretm + phi
			}

			temp = z.Get(ii-1) / delta.Get(ii-1)
			dw = dpsi + dphi + temp*temp
			temp = z.Get(ii-1) * temp
			w = rhoinv + phi + psi + temp
			erretm = eight*(phi-psi) + erretm + two*rhoinv + three*math.Abs(temp) + math.Abs(tau)*dw
			if w*prew > zero && math.Abs(w) > math.Abs(prew)/ten {
				swtch = !swtch
			}

		}

		//        Return with INFO = 1, NITER = MAXIT and not converged
		(*info) = 1
		if orgati {
			(*dlam) = d.Get((*i)-1) + tau
		} else {
			(*dlam) = d.Get(ip1-1) + tau
		}

	}
}
