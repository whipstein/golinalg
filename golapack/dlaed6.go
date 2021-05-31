package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlaed6 computes the positive or negative root (closest to the origin)
// of
//                  z(1)        z(2)        z(3)
// f(x) =   rho + --------- + ---------- + ---------
//                 d(1)-x      d(2)-x      d(3)-x
//
// It is assumed that
//
//       if ORGATI = .true. the root is between d(2) and d(3);
//       otherwise it is between d(1) and d(2)
//
// This routine will be called by DLAED4 when necessary. In most cases,
// the root sought is the smallest in magnitude, though it might not be
// in some extremely rare situations.
func Dlaed6(kniter *int, orgati bool, rho *float64, d, z *mat.Vector, finit, tau *float64, info *int) {
	var scale bool
	var a, b, base, c, ddf, df, eight, eps, erretm, eta, f, fc, four, lbd, one, sclfac, sclinv, small1, small2, sminv1, sminv2, temp, temp1, temp2, temp3, temp4, three, two, ubd, zero float64
	var i, iter, maxit, niter int
	dscale := vf(3)
	zscale := vf(3)

	maxit = 40
	zero = 0.0
	one = 1.0
	two = 2.0
	three = 3.0
	four = 4.0
	eight = 8.0

	(*info) = 0

	if orgati {
		lbd = d.Get(1)
		ubd = d.Get(2)
	} else {
		lbd = d.Get(0)
		ubd = d.Get(1)
	}
	if (*finit) < zero {
		lbd = zero
	} else {
		ubd = zero
	}

	niter = 1
	(*tau) = zero
	if (*kniter) == 2 {
		if orgati {
			temp = (d.Get(2) - d.Get(1)) / two
			c = (*rho) + z.Get(0)/((d.Get(0)-d.Get(1))-temp)
			a = c*(d.Get(1)+d.Get(2)) + z.Get(1) + z.Get(2)
			b = c*d.Get(1)*d.Get(2) + z.Get(1)*d.Get(2) + z.Get(2)*d.Get(1)
		} else {
			temp = (d.Get(0) - d.Get(1)) / two
			c = (*rho) + z.Get(2)/((d.Get(2)-d.Get(1))-temp)
			a = c*(d.Get(0)+d.Get(1)) + z.Get(0) + z.Get(1)
			b = c*d.Get(0)*d.Get(1) + z.Get(0)*d.Get(1) + z.Get(1)*d.Get(0)
		}
		temp = maxf64(math.Abs(a), math.Abs(b), math.Abs(c))
		a = a / temp
		b = b / temp
		c = c / temp
		if c == zero {
			(*tau) = b / a
		} else if a <= zero {
			(*tau) = (a - math.Sqrt(math.Abs(a*a-four*b*c))) / (two * c)
		} else {
			(*tau) = two * b / (a + math.Sqrt(math.Abs(a*a-four*b*c)))
		}
		if (*tau) < lbd || (*tau) > ubd {
			(*tau) = (lbd + ubd) / two
		}
		if d.Get(0) == (*tau) || d.Get(1) == (*tau) || d.Get(2) == (*tau) {
			(*tau) = zero
		} else {
			temp = (*finit) + (*tau)*z.Get(0)/(d.Get(0)*(d.Get(0)-(*tau))) + (*tau)*z.Get(1)/(d.Get(1)*(d.Get(1)-(*tau))) + (*tau)*z.Get(2)/(d.Get(2)*(d.Get(2)-(*tau)))
			if temp <= zero {
				lbd = (*tau)
			} else {
				ubd = (*tau)
			}
			if math.Abs(*finit) <= math.Abs(temp) {
				(*tau) = zero
			}
		}
	}

	//     get machine parameters for possible scaling to avoid overflow
	//
	//     modified by Sven: parameters SMALL1, SMINV1, SMALL2,
	//     SMINV2, EPS are not SAVEd anymore between one call to the
	//     others but recomputed at each call
	eps = Dlamch(Epsilon)
	base = Dlamch(Base)
	small1 = math.Pow(base, float64(int(math.Log(Dlamch(SafeMinimum))/math.Log(base)/three)))
	sminv1 = one / small1
	small2 = small1 * small1
	sminv2 = sminv1 * sminv1

	//     Determine if scaling of inputs necessary to avoid overflow
	//     when computing 1/TEMP**3
	if orgati {
		temp = minf64(math.Abs(d.Get(1)-(*tau)), math.Abs(d.Get(2)-(*tau)))
	} else {
		temp = minf64(math.Abs(d.Get(0)-(*tau)), math.Abs(d.Get(1)-(*tau)))
	}
	scale = false
	if temp <= small1 {
		scale = true
		if temp <= small2 {
			//        Scale up by power of radix nearest 1/SAFMIN**(2/3)
			sclfac = sminv2
			sclinv = small2
		} else {
			//        Scale up by power of radix nearest 1/SAFMIN**(1/3)
			sclfac = sminv1
			sclinv = small1
		}

		//        Scaling up safe because D, Z, TAU scaled elsewhere to be O(1)
		for i = 1; i <= 3; i++ {
			dscale.Set(i-1, d.Get(i-1)*sclfac)
			zscale.Set(i-1, z.Get(i-1)*sclfac)
		}
		(*tau) = (*tau) * sclfac
		lbd = lbd * sclfac
		ubd = ubd * sclfac
	} else {
		//        Copy D and Z to DSCALE and ZSCALE
		for i = 1; i <= 3; i++ {
			dscale.Set(i-1, d.Get(i-1))
			zscale.Set(i-1, z.Get(i-1))
		}
	}

	fc = zero
	df = zero
	ddf = zero
	for i = 1; i <= 3; i++ {
		temp = one / (dscale.Get(i-1) - (*tau))
		temp1 = zscale.Get(i-1) * temp
		temp2 = temp1 * temp
		temp3 = temp2 * temp
		fc = fc + temp1/dscale.Get(i-1)
		df = df + temp2
		ddf = ddf + temp3
	}
	f = (*finit) + (*tau)*fc

	if math.Abs(f) <= zero {
		goto label60
	}
	if f <= zero {
		lbd = (*tau)
	} else {
		ubd = (*tau)
	}

	//        Iteration begins -- Use Gragg-Thornton-Warner cubic convergent
	//                            scheme
	//
	//     It is not hard to see that
	//
	//           1) Iterations will go up monotonically
	//              if FINIT < 0;
	//
	//           2) Iterations will go down monotonically
	//              if FINIT > 0.
	iter = niter + 1

	for niter = iter; niter <= maxit; niter++ {

		if orgati {
			temp1 = dscale.Get(1) - (*tau)
			temp2 = dscale.Get(2) - (*tau)
		} else {
			temp1 = dscale.Get(0) - (*tau)
			temp2 = dscale.Get(1) - (*tau)
		}
		a = (temp1+temp2)*f - temp1*temp2*df
		b = temp1 * temp2 * f
		c = f - (temp1+temp2)*df + temp1*temp2*ddf
		temp = maxf64(math.Abs(a), math.Abs(b), math.Abs(c))
		a = a / temp
		b = b / temp
		c = c / temp
		if c == zero {
			eta = b / a
		} else if a <= zero {
			eta = (a - math.Sqrt(math.Abs(a*a-four*b*c))) / (two * c)
		} else {
			eta = two * b / (a + math.Sqrt(math.Abs(a*a-four*b*c)))
		}
		if f*eta >= zero {
			eta = -f / df
		}

		(*tau) = (*tau) + eta
		if (*tau) < lbd || (*tau) > ubd {
			(*tau) = (lbd + ubd) / two
		}

		fc = zero
		erretm = zero
		df = zero
		ddf = zero
		for i = 1; i <= 3; i++ {
			if (dscale.Get(i-1) - (*tau)) != zero {
				temp = one / (dscale.Get(i-1) - (*tau))
				temp1 = zscale.Get(i-1) * temp
				temp2 = temp1 * temp
				temp3 = temp2 * temp
				temp4 = temp1 / dscale.Get(i-1)
				fc = fc + temp4
				erretm = erretm + math.Abs(temp4)
				df = df + temp2
				ddf = ddf + temp3
			} else {
				goto label60
			}
		}
		f = (*finit) + (*tau)*fc
		erretm = eight*(math.Abs(*finit)+math.Abs(*tau)*erretm) + math.Abs(*tau)*df
		if (math.Abs(f) <= four*eps*erretm) || ((ubd - lbd) <= four*eps*math.Abs(*tau)) {
			goto label60
		}
		if f <= zero {
			lbd = (*tau)
		} else {
			ubd = (*tau)
		}
	}
	(*info) = 1
label60:
	;

	//     Undo scaling
	if scale {
		(*tau) = (*tau) * sclinv
	}
}
