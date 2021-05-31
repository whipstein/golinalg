package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlasd4 subroutine computes the square root of the I-th updated
// eigenvalue of a positive symmetric rank-one modification to
// a positive diagonal matrix whose entries are given as the squares
// of the corresponding entries in the array d, and that
//
//        0 <= D(i) < D(j)  for  i < j
//
// and that RHO > 0. This is arranged by the calling routine, and is
// no loss in generality.  The rank-one modified system is thus
//
//        diag( D ) * diag( D ) +  RHO * Z * Z_transpose.
//
// where we assume the Euclidean norm of Z is 1.
//
// The method consists of approximating the rational functions in the
// secular equation by simpler interpolating rational functions.
func Dlasd4(n, i *int, d, z, delta *mat.Vector, rho, sigma *float64, work *mat.Vector, info *int) {
	var geomavg, orgati, swtch, swtch3 bool
	var a, b, c, delsq, delsq2, dphi, dpsi, dtiim, dtiip, dtipsq, dtisq, dtnsq, dtnsq1, dw, eight, eps, erretm, eta, four, one, phi, prew, psi, rhoinv, sglb, sgub, sq2, tau, tau2, temp, temp1, temp2, ten, three, two, w, zero float64
	var ii, iim1, iip1, ip1, iter, j, maxit, niter int
	dd := vf(3)
	zz := vf(3)

	maxit = 400
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
		//        Presumably, I=1 upon entry
		(*sigma) = math.Sqrt(d.Get(0)*d.Get(0) + (*rho)*z.Get(0)*z.Get(0))
		delta.Set(0, one)
		work.Set(0, one)
		return
	}
	if (*n) == 2 {
		Dlasd5(i, d, z, delta, rho, sigma, work)
		return
	}

	//     Compute machine epsilon
	eps = Dlamch(Epsilon)
	rhoinv = one / (*rho)
	tau2 = zero

	//     The case I = N
	if (*i) == (*n) {
		//        Initialize some basic variables
		ii = (*n) - 1
		niter = 1

		//        Calculate initial guess
		temp = (*rho) / two

		//        If ||Z||_2 is not one, then TEMP should be set to
		//        RHO * ||Z||_2^2 / TWO
		temp1 = temp / (d.Get((*n)-1) + math.Sqrt(d.Get((*n)-1)*d.Get((*n)-1)+temp))
		for j = 1; j <= (*n); j++ {
			work.Set(j-1, d.Get(j-1)+d.Get((*n)-1)+temp1)
			delta.Set(j-1, (d.Get(j-1)-d.Get((*n)-1))-temp1)
		}

		psi = zero
		for j = 1; j <= (*n)-2; j++ {
			psi = psi + z.Get(j-1)*z.Get(j-1)/(delta.Get(j-1)*work.Get(j-1))
		}

		c = rhoinv + psi
		w = c + z.Get(ii-1)*z.Get(ii-1)/(delta.Get(ii-1)*work.Get(ii-1)) + z.Get((*n)-1)*z.Get((*n)-1)/(delta.Get((*n)-1)*work.Get((*n)-1))

		if w <= zero {
			temp1 = math.Sqrt(d.Get((*n)-1)*d.Get((*n)-1) + (*rho))
			temp = z.Get((*n)-1-1)*z.Get((*n)-1-1)/((d.Get((*n)-1-1)+temp1)*(d.Get((*n)-1)-d.Get((*n)-1-1)+(*rho)/(d.Get((*n)-1)+temp1))) + z.Get((*n)-1)*z.Get((*n)-1)/(*rho)

			//           The following TAU2 is to approximate
			//           SIGMA_n^2 - D( N )*D( N )
			if c <= temp {
				tau = (*rho)
			} else {
				delsq = (d.Get((*n)-1) - d.Get((*n)-1-1)) * (d.Get((*n)-1) + d.Get((*n)-1-1))
				a = -c*delsq + z.Get((*n)-1-1)*z.Get((*n)-1-1) + z.Get((*n)-1)*z.Get((*n)-1)
				b = z.Get((*n)-1) * z.Get((*n)-1) * delsq
				if a < zero {
					tau2 = two * b / (math.Sqrt(a*a+four*b*c) - a)
				} else {
					tau2 = (a + math.Sqrt(a*a+four*b*c)) / (two * c)
				}
				tau = tau2 / (d.Get((*n)-1) + math.Sqrt(d.Get((*n)-1)*d.Get((*n)-1)+tau2))
			}

			//           It can be proved that
			//               D(N)^2+RHO/2 <= SIGMA_n^2 < D(N)^2+TAU2 <= D(N)^2+RHO
		} else {
			delsq = (d.Get((*n)-1) - d.Get((*n)-1-1)) * (d.Get((*n)-1) + d.Get((*n)-1-1))
			a = -c*delsq + z.Get((*n)-1-1)*z.Get((*n)-1-1) + z.Get((*n)-1)*z.Get((*n)-1)
			b = z.Get((*n)-1) * z.Get((*n)-1) * delsq

			//           The following TAU2 is to approximate
			//           SIGMA_n^2 - D( N )*D( N )
			if a < zero {
				tau2 = two * b / (math.Sqrt(a*a+four*b*c) - a)
			} else {
				tau2 = (a + math.Sqrt(a*a+four*b*c)) / (two * c)
			}
			tau = tau2 / (d.Get((*n)-1) + math.Sqrt(d.Get((*n)-1)*d.Get((*n)-1)+tau2))

			//           It can be proved that
			//           D(N)^2 < D(N)^2+TAU2 < SIGMA(N)^2 < D(N)^2+RHO/2
		}

		//        The following TAU is to approximate SIGMA_n - D( N )
		//
		//         TAU = TAU2 / ( D( N )+SQRT( D( N )*D( N )+TAU2 ) )
		(*sigma) = d.Get((*n)-1) + tau
		for j = 1; j <= (*n); j++ {
			delta.Set(j-1, (d.Get(j-1)-d.Get((*n)-1))-tau)
			work.Set(j-1, d.Get(j-1)+d.Get((*n)-1)+tau)
		}

		//        Evaluate PSI and the derivative DPSI
		dpsi = zero
		psi = zero
		erretm = zero
		for j = 1; j <= ii; j++ {
			temp = z.Get(j-1) / (delta.Get(j-1) * work.Get(j-1))
			psi = psi + z.Get(j-1)*temp
			dpsi = dpsi + temp*temp
			erretm = erretm + psi
		}
		erretm = math.Abs(erretm)

		//        Evaluate PHI and the derivative DPHI
		temp = z.Get((*n)-1) / (delta.Get((*n)-1) * work.Get((*n)-1))
		phi = z.Get((*n)-1) * temp
		dphi = temp * temp
		erretm = eight*(-phi-psi) + erretm - phi + rhoinv
		//    $          + ABS( TAU2 )*( DPSI+DPHI )

		w = rhoinv + phi + psi

		//        Test for convergence
		if math.Abs(w) <= eps*erretm {
			return
		}

		//        Calculate the new step
		niter = niter + 1
		dtnsq1 = work.Get((*n)-1-1) * delta.Get((*n)-1-1)
		dtnsq = work.Get((*n)-1) * delta.Get((*n)-1)
		c = w - dtnsq1*dpsi - dtnsq*dphi
		a = (dtnsq+dtnsq1)*w - dtnsq*dtnsq1*(dpsi+dphi)
		b = dtnsq * dtnsq1 * w
		if c < zero {
			c = math.Abs(c)
		}
		if c == zero {
			eta = (*rho) - (*sigma)*(*sigma)
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
		temp = eta - dtnsq
		if temp > (*rho) {
			eta = (*rho) + dtnsq
		}

		eta = eta / ((*sigma) + math.Sqrt(eta+(*sigma)*(*sigma)))
		tau = tau + eta
		(*sigma) = (*sigma) + eta

		for j = 1; j <= (*n); j++ {
			delta.Set(j-1, delta.Get(j-1)-eta)
			work.Set(j-1, work.Get(j-1)+eta)
		}

		//        Evaluate PSI and the derivative DPSI
		dpsi = zero
		psi = zero
		erretm = zero
		for j = 1; j <= ii; j++ {
			temp = z.Get(j-1) / (work.Get(j-1) * delta.Get(j-1))
			psi = psi + z.Get(j-1)*temp
			dpsi = dpsi + temp*temp
			erretm = erretm + psi
		}
		erretm = math.Abs(erretm)

		//        Evaluate PHI and the derivative DPHI
		tau2 = work.Get((*n)-1) * delta.Get((*n)-1)
		temp = z.Get((*n)-1) / tau2
		phi = z.Get((*n)-1) * temp
		dphi = temp * temp
		erretm = eight*(-phi-psi) + erretm - phi + rhoinv
		//    $          + ABS( TAU2 )*( DPSI+DPHI )

		w = rhoinv + phi + psi

		//        Main loop to update the values of the array   DELTA
		iter = niter + 1

		for niter = iter; niter <= maxit; niter++ {
			//           Test for convergence
			if math.Abs(w) <= eps*erretm {
				return
			}

			//           Calculate the new step
			dtnsq1 = work.Get((*n)-1-1) * delta.Get((*n)-1-1)
			dtnsq = work.Get((*n)-1) * delta.Get((*n)-1)
			c = w - dtnsq1*dpsi - dtnsq*dphi
			a = (dtnsq+dtnsq1)*w - dtnsq1*dtnsq*(dpsi+dphi)
			b = dtnsq1 * dtnsq * w
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
			temp = eta - dtnsq
			if temp <= zero {
				eta = eta / two
			}

			eta = eta / ((*sigma) + math.Sqrt(eta+(*sigma)*(*sigma)))
			tau = tau + eta
			(*sigma) = (*sigma) + eta

			for j = 1; j <= (*n); j++ {
				delta.Set(j-1, delta.Get(j-1)-eta)
				work.Set(j-1, work.Get(j-1)+eta)
			}

			//           Evaluate PSI and the derivative DPSI
			dpsi = zero
			psi = zero
			erretm = zero
			for j = 1; j <= ii; j++ {
				temp = z.Get(j-1) / (work.Get(j-1) * delta.Get(j-1))
				psi = psi + z.Get(j-1)*temp
				dpsi = dpsi + temp*temp
				erretm = erretm + psi
			}
			erretm = math.Abs(erretm)

			//           Evaluate PHI and the derivative DPHI
			tau2 = work.Get((*n)-1) * delta.Get((*n)-1)
			temp = z.Get((*n)-1) / tau2
			phi = z.Get((*n)-1) * temp
			dphi = temp * temp
			erretm = eight*(-phi-psi) + erretm - phi + rhoinv
			//    $             + ABS( TAU2 )*( DPSI+DPHI )

			w = rhoinv + phi + psi
		}

		//        Return with INFO = 1, NITER = MAXIT and not converged
		(*info) = 1
		return

		//        End for the case I = N
	} else {
		//        The case for I < N
		niter = 1
		ip1 = (*i) + 1

		//        Calculate initial guess
		delsq = (d.Get(ip1-1) - d.Get((*i)-1)) * (d.Get(ip1-1) + d.Get((*i)-1))
		delsq2 = delsq / two
		sq2 = math.Sqrt((d.Get((*i)-1)*d.Get((*i)-1) + d.Get(ip1-1)*d.Get(ip1-1)) / two)
		temp = delsq2 / (d.Get((*i)-1) + sq2)
		for j = 1; j <= (*n); j++ {
			work.Set(j-1, d.Get(j-1)+d.Get((*i)-1)+temp)
			delta.Set(j-1, (d.Get(j-1)-d.Get((*i)-1))-temp)
		}

		psi = zero
		for j = 1; j <= (*i)-1; j++ {
			psi = psi + z.Get(j-1)*z.Get(j-1)/(work.Get(j-1)*delta.Get(j-1))
		}

		phi = zero
		for j = (*n); j >= (*i)+2; j-- {
			phi = phi + z.Get(j-1)*z.Get(j-1)/(work.Get(j-1)*delta.Get(j-1))
		}
		c = rhoinv + psi + phi
		w = c + z.Get((*i)-1)*z.Get((*i)-1)/(work.Get((*i)-1)*delta.Get((*i)-1)) + z.Get(ip1-1)*z.Get(ip1-1)/(work.Get(ip1-1)*delta.Get(ip1-1))

		geomavg = false
		if w > zero {
			//           d(i)^2 < the ith sigma^2 < (d(i)^2+d(i+1)^2)/2
			//
			//           We choose d(i) as origin.
			orgati = true
			ii = (*i)
			sglb = zero
			sgub = delsq2 / (d.Get((*i)-1) + sq2)
			a = c*delsq + z.Get((*i)-1)*z.Get((*i)-1) + z.Get(ip1-1)*z.Get(ip1-1)
			b = z.Get((*i)-1) * z.Get((*i)-1) * delsq
			if a > zero {
				tau2 = two * b / (a + math.Sqrt(math.Abs(a*a-four*b*c)))
			} else {
				tau2 = (a - math.Sqrt(math.Abs(a*a-four*b*c))) / (two * c)
			}

			//           TAU2 now is an estimation of SIGMA^2 - D( I )^2. The
			//           following, however, is the corresponding estimation of
			//           SIGMA - D( I ).
			tau = tau2 / (d.Get((*i)-1) + math.Sqrt(d.Get((*i)-1)*d.Get((*i)-1)+tau2))
			temp = math.Sqrt(eps)
			if (d.Get((*i)-1) <= temp*d.Get(ip1-1)) && (math.Abs(z.Get((*i)-1)) <= temp) && (d.Get((*i)-1) > zero) {
				tau = minf64(ten*d.Get((*i)-1), sgub)
				geomavg = true
			}
		} else {
			//           (d(i)^2+d(i+1)^2)/2 <= the ith sigma^2 < d(i+1)^2/2
			//
			//           We choose d(i+1) as origin.
			orgati = false
			ii = ip1
			sglb = -delsq2 / (d.Get(ii-1) + sq2)
			sgub = zero
			a = c*delsq - z.Get((*i)-1)*z.Get((*i)-1) - z.Get(ip1-1)*z.Get(ip1-1)
			b = z.Get(ip1-1) * z.Get(ip1-1) * delsq
			if a < zero {
				tau2 = two * b / (a - math.Sqrt(math.Abs(a*a+four*b*c)))
			} else {
				tau2 = -(a + math.Sqrt(math.Abs(a*a+four*b*c))) / (two * c)
			}

			//           TAU2 now is an estimation of SIGMA^2 - D( IP1 )^2. The
			//           following, however, is the corresponding estimation of
			//           SIGMA - D( IP1 ).
			tau = tau2 / (d.Get(ip1-1) + math.Sqrt(math.Abs(d.Get(ip1-1)*d.Get(ip1-1)+tau2)))
		}

		(*sigma) = d.Get(ii-1) + tau
		for j = 1; j <= (*n); j++ {
			work.Set(j-1, d.Get(j-1)+d.Get(ii-1)+tau)
			delta.Set(j-1, (d.Get(j-1)-d.Get(ii-1))-tau)
		}
		iim1 = ii - 1
		iip1 = ii + 1

		//        Evaluate PSI and the derivative DPSI
		dpsi = zero
		psi = zero
		erretm = zero
		for j = 1; j <= iim1; j++ {
			temp = z.Get(j-1) / (work.Get(j-1) * delta.Get(j-1))
			psi = psi + z.Get(j-1)*temp
			dpsi = dpsi + temp*temp
			erretm = erretm + psi
		}
		erretm = math.Abs(erretm)

		//        Evaluate PHI and the derivative DPHI
		dphi = zero
		phi = zero
		for j = (*n); j >= iip1; j-- {
			temp = z.Get(j-1) / (work.Get(j-1) * delta.Get(j-1))
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

		temp = z.Get(ii-1) / (work.Get(ii-1) * delta.Get(ii-1))
		dw = dpsi + dphi + temp*temp
		temp = z.Get(ii-1) * temp
		w = w + temp
		erretm = eight*(phi-psi) + erretm + two*rhoinv + three*math.Abs(temp)
		//    $          + ABS( TAU2 )*DW

		//        Test for convergence
		if math.Abs(w) <= eps*erretm {
			return
		}

		if w <= zero {
			sglb = maxf64(sglb, tau)
		} else {
			sgub = minf64(sgub, tau)
		}

		//        Calculate the new step
		niter = niter + 1
		if !swtch3 {
			dtipsq = work.Get(ip1-1) * delta.Get(ip1-1)
			dtisq = work.Get((*i)-1) * delta.Get((*i)-1)
			if orgati {
				c = w - dtipsq*dw + delsq*math.Pow(z.Get((*i)-1)/dtisq, 2)
			} else {
				c = w - dtisq*dw - delsq*math.Pow(z.Get(ip1-1)/dtipsq, 2)
			}
			a = (dtipsq+dtisq)*w - dtipsq*dtisq*dw
			b = dtipsq * dtisq * w
			if c == zero {
				if a == zero {
					if orgati {
						a = z.Get((*i)-1)*z.Get((*i)-1) + dtipsq*dtipsq*(dpsi+dphi)
					} else {
						a = z.Get(ip1-1)*z.Get(ip1-1) + dtisq*dtisq*(dpsi+dphi)
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
			dtiim = work.Get(iim1-1) * delta.Get(iim1-1)
			dtiip = work.Get(iip1-1) * delta.Get(iip1-1)
			temp = rhoinv + psi + phi
			if orgati {
				temp1 = z.Get(iim1-1) / dtiim
				temp1 = temp1 * temp1
				c = (temp - dtiip*(dpsi+dphi)) - (d.Get(iim1-1)-d.Get(iip1-1))*(d.Get(iim1-1)+d.Get(iip1-1))*temp1
				zz.Set(0, z.Get(iim1-1)*z.Get(iim1-1))
				if dpsi < temp1 {
					zz.Set(2, dtiip*dtiip*dphi)
				} else {
					zz.Set(2, dtiip*dtiip*((dpsi-temp1)+dphi))
				}
			} else {
				temp1 = z.Get(iip1-1) / dtiip
				temp1 = temp1 * temp1
				c = (temp - dtiim*(dpsi+dphi)) - (d.Get(iip1-1)-d.Get(iim1-1))*(d.Get(iim1-1)+d.Get(iip1-1))*temp1
				if dphi < temp1 {
					zz.Set(0, dtiim*dtiim*dpsi)
				} else {
					zz.Set(0, dtiim*dtiim*(dpsi+(dphi-temp1)))
				}
				zz.Set(2, z.Get(iip1-1)*z.Get(iip1-1))
			}
			zz.Set(1, z.Get(ii-1)*z.Get(ii-1))
			dd.Set(0, dtiim)
			dd.Set(1, delta.Get(ii-1)*work.Get(ii-1))
			dd.Set(2, dtiip)
			Dlaed6(&niter, orgati, &c, dd, zz, &w, &eta, info)

			if (*info) != 0 {
				//              If INFO is not 0, i.e., DLAED6 failed, switch back
				//              to 2 pole interpolation.
				swtch3 = false
				(*info) = 0
				dtipsq = work.Get(ip1-1) * delta.Get(ip1-1)
				dtisq = work.Get((*i)-1) * delta.Get((*i)-1)
				if orgati {
					c = w - dtipsq*dw + delsq*math.Pow(z.Get((*i)-1)/dtisq, 2)
				} else {
					c = w - dtisq*dw - delsq*math.Pow(z.Get(ip1-1)/dtipsq, 2)
				}
				a = (dtipsq+dtisq)*w - dtipsq*dtisq*dw
				b = dtipsq * dtisq * w
				if c == zero {
					if a == zero {
						if orgati {
							a = z.Get((*i)-1)*z.Get((*i)-1) + dtipsq*dtipsq*(dpsi+dphi)
						} else {
							a = z.Get(ip1-1)*z.Get(ip1-1) + dtisq*dtisq*(dpsi+dphi)
						}
					}
					eta = b / a
				} else if a <= zero {
					eta = (a - math.Sqrt(math.Abs(a*a-four*b*c))) / (two * c)
				} else {
					eta = two * b / (a + math.Sqrt(math.Abs(a*a-four*b*c)))
				}
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

		eta = eta / ((*sigma) + math.Sqrt((*sigma)*(*sigma)+eta))
		temp = tau + eta
		if temp > sgub || temp < sglb {
			if w < zero {
				eta = (sgub - tau) / two
			} else {
				eta = (sglb - tau) / two
			}
			if geomavg {
				if w < zero {
					if tau > zero {
						eta = math.Sqrt(sgub*tau) - tau
					}
				} else {
					if sglb > zero {
						eta = math.Sqrt(sglb*tau) - tau
					}
				}
			}
		}

		prew = w

		tau = tau + eta
		(*sigma) = (*sigma) + eta

		for j = 1; j <= (*n); j++ {
			work.Set(j-1, work.Get(j-1)+eta)
			delta.Set(j-1, delta.Get(j-1)-eta)
		}

		//        Evaluate PSI and the derivative DPSI
		dpsi = zero
		psi = zero
		erretm = zero
		for j = 1; j <= iim1; j++ {
			temp = z.Get(j-1) / (work.Get(j-1) * delta.Get(j-1))
			psi = psi + z.Get(j-1)*temp
			dpsi = dpsi + temp*temp
			erretm = erretm + psi
		}
		erretm = math.Abs(erretm)

		//        Evaluate PHI and the derivative DPHI
		dphi = zero
		phi = zero
		for j = (*n); j >= iip1; j-- {
			temp = z.Get(j-1) / (work.Get(j-1) * delta.Get(j-1))
			phi = phi + z.Get(j-1)*temp
			dphi = dphi + temp*temp
			erretm = erretm + phi
		}

		tau2 = work.Get(ii-1) * delta.Get(ii-1)
		temp = z.Get(ii-1) / tau2
		dw = dpsi + dphi + temp*temp
		temp = z.Get(ii-1) * temp
		w = rhoinv + phi + psi + temp
		erretm = eight*(phi-psi) + erretm + two*rhoinv + three*math.Abs(temp)
		//    $          + ABS( TAU2 )*DW

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

		//        Main loop to update the values of the array   DELTA and WORK
		iter = niter + 1

		for niter = iter; niter <= maxit; niter++ {
			//           Test for convergence
			if math.Abs(w) <= eps*erretm {
				//     $          .OR. (SGUB-SGLB).LE.EIGHT*ABS(SGUB+SGLB) ) THEN
				return
			}

			if w <= zero {
				sglb = maxf64(sglb, tau)
			} else {
				sgub = minf64(sgub, tau)
			}

			//           Calculate the new step
			if !swtch3 {
				dtipsq = work.Get(ip1-1) * delta.Get(ip1-1)
				dtisq = work.Get((*i)-1) * delta.Get((*i)-1)
				if !swtch {
					if orgati {
						c = w - dtipsq*dw + delsq*math.Pow(z.Get((*i)-1)/dtisq, 2)
					} else {
						c = w - dtisq*dw - delsq*math.Pow(z.Get(ip1-1)/dtipsq, 2)
					}
				} else {
					temp = z.Get(ii-1) / (work.Get(ii-1) * delta.Get(ii-1))
					if orgati {
						dpsi = dpsi + temp*temp
					} else {
						dphi = dphi + temp*temp
					}
					c = w - dtisq*dpsi - dtipsq*dphi
				}
				a = (dtipsq+dtisq)*w - dtipsq*dtisq*dw
				b = dtipsq * dtisq * w
				if c == zero {
					if a == zero {
						if !swtch {
							if orgati {
								a = z.Get((*i)-1)*z.Get((*i)-1) + dtipsq*dtipsq*(dpsi+dphi)
							} else {
								a = z.Get(ip1-1)*z.Get(ip1-1) + dtisq*dtisq*(dpsi+dphi)
							}
						} else {
							a = dtisq*dtisq*dpsi + dtipsq*dtipsq*dphi
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
				dtiim = work.Get(iim1-1) * delta.Get(iim1-1)
				dtiip = work.Get(iip1-1) * delta.Get(iip1-1)
				temp = rhoinv + psi + phi
				if swtch {
					c = temp - dtiim*dpsi - dtiip*dphi
					zz.Set(0, dtiim*dtiim*dpsi)
					zz.Set(2, dtiip*dtiip*dphi)
				} else {
					if orgati {
						temp1 = z.Get(iim1-1) / dtiim
						temp1 = temp1 * temp1
						temp2 = (d.Get(iim1-1) - d.Get(iip1-1)) * (d.Get(iim1-1) + d.Get(iip1-1)) * temp1
						c = temp - dtiip*(dpsi+dphi) - temp2
						zz.Set(0, z.Get(iim1-1)*z.Get(iim1-1))
						if dpsi < temp1 {
							zz.Set(2, dtiip*dtiip*dphi)
						} else {
							zz.Set(2, dtiip*dtiip*((dpsi-temp1)+dphi))
						}
					} else {
						temp1 = z.Get(iip1-1) / dtiip
						temp1 = temp1 * temp1
						temp2 = (d.Get(iip1-1) - d.Get(iim1-1)) * (d.Get(iim1-1) + d.Get(iip1-1)) * temp1
						c = temp - dtiim*(dpsi+dphi) - temp2
						if dphi < temp1 {
							zz.Set(0, dtiim*dtiim*dpsi)
						} else {
							zz.Set(0, dtiim*dtiim*(dpsi+(dphi-temp1)))
						}
						zz.Set(2, z.Get(iip1-1)*z.Get(iip1-1))
					}
				}
				dd.Set(0, dtiim)
				dd.Set(1, delta.Get(ii-1)*work.Get(ii-1))
				dd.Set(2, dtiip)
				Dlaed6(&niter, orgati, &c, dd, zz, &w, &eta, info)

				if (*info) != 0 {
					//                 If INFO is not 0, i.e., DLAED6 failed, switch
					//                 back to two pole interpolation
					swtch3 = false
					(*info) = 0
					dtipsq = work.Get(ip1-1) * delta.Get(ip1-1)
					dtisq = work.Get((*i)-1) * delta.Get((*i)-1)
					if !swtch {
						if orgati {
							c = w - dtipsq*dw + delsq*math.Pow(z.Get((*i)-1)/dtisq, 2)
						} else {
							c = w - dtisq*dw - delsq*math.Pow(z.Get(ip1-1)/dtipsq, 2)
						}
					} else {
						temp = z.Get(ii-1) / (work.Get(ii-1) * delta.Get(ii-1))
						if orgati {
							dpsi = dpsi + temp*temp
						} else {
							dphi = dphi + temp*temp
						}
						c = w - dtisq*dpsi - dtipsq*dphi
					}
					a = (dtipsq+dtisq)*w - dtipsq*dtisq*dw
					b = dtipsq * dtisq * w
					if c == zero {
						if a == zero {
							if !swtch {
								if orgati {
									a = z.Get((*i)-1)*z.Get((*i)-1) + dtipsq*dtipsq*(dpsi+dphi)
								} else {
									a = z.Get(ip1-1)*z.Get(ip1-1) + dtisq*dtisq*(dpsi+dphi)
								}
							} else {
								a = dtisq*dtisq*dpsi + dtipsq*dtipsq*dphi
							}
						}
						eta = b / a
					} else if a <= zero {
						eta = (a - math.Sqrt(math.Abs(a*a-four*b*c))) / (two * c)
					} else {
						eta = two * b / (a + math.Sqrt(math.Abs(a*a-four*b*c)))
					}
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

			eta = eta / ((*sigma) + math.Sqrt((*sigma)*(*sigma)+eta))
			temp = tau + eta
			if temp > sgub || temp < sglb {
				if w < zero {
					eta = (sgub - tau) / two
				} else {
					eta = (sglb - tau) / two
				}
				if geomavg {
					if w < zero {
						if tau > zero {
							eta = math.Sqrt(sgub*tau) - tau
						}
					} else {
						if sglb > zero {
							eta = math.Sqrt(sglb*tau) - tau
						}
					}
				}
			}

			prew = w

			tau = tau + eta
			(*sigma) = (*sigma) + eta

			for j = 1; j <= (*n); j++ {
				work.Set(j-1, work.Get(j-1)+eta)
				delta.Set(j-1, delta.Get(j-1)-eta)
			}

			//           Evaluate PSI and the derivative DPSI
			dpsi = zero
			psi = zero
			erretm = zero
			for j = 1; j <= iim1; j++ {
				temp = z.Get(j-1) / (work.Get(j-1) * delta.Get(j-1))
				psi = psi + z.Get(j-1)*temp
				dpsi = dpsi + temp*temp
				erretm = erretm + psi
			}
			erretm = math.Abs(erretm)

			//           Evaluate PHI and the derivative DPHI
			dphi = zero
			phi = zero
			for j = (*n); j >= iip1; j-- {
				temp = z.Get(j-1) / (work.Get(j-1) * delta.Get(j-1))
				phi = phi + z.Get(j-1)*temp
				dphi = dphi + temp*temp
				erretm = erretm + phi
			}

			tau2 = work.Get(ii-1) * delta.Get(ii-1)
			temp = z.Get(ii-1) / tau2
			dw = dpsi + dphi + temp*temp
			temp = z.Get(ii-1) * temp
			w = rhoinv + phi + psi + temp
			erretm = eight*(phi-psi) + erretm + two*rhoinv + three*math.Abs(temp)
			//    $             + ABS( TAU2 )*DW

			if w*prew > zero && math.Abs(w) > math.Abs(prew)/ten {
				swtch = !swtch
			}

		}

		//        Return with INFO = 1, NITER = MAXIT and not converged
		(*info) = 1

	}
}
