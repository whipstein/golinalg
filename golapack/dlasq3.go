package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlasq3 checks for deflation, computes a shift (TAU) and calls dqds.
// In case of failure it changes shifts, and tries again until output
// is positive.
func Dlasq3(i0, n0 *int, z *mat.Vector, pp *int, dmin, sigma, desig, qmax *float64, nfail, iter, ndiv *int, ieee *bool, ttype *int, dmin1, dmin2, dn, dn1, dn2, g, tau *float64) {
	var cbias, eps, half, hundrd, one, qurtr, s, t, temp, tol, tol2, two, zero float64
	var ipn4, j4, n0in, nn int

	cbias = 1.50
	zero = 0.0
	qurtr = 0.250
	half = 0.5
	one = 1.0
	two = 2.0
	hundrd = 100.0

	n0in = (*n0)
	eps = Dlamch(Precision)
	tol = eps * hundrd
	tol2 = math.Pow(tol, 2)

	//     Check for deflation.
label10:
	;

	if (*n0) < (*i0) {
		return
	}
	if (*n0) == (*i0) {
		goto label20
	}
	nn = 4*(*n0) + (*pp)
	if (*n0) == ((*i0) + 1) {
		goto label40
	}

	//     Check whether E(N0-1) is negligible, 1 eigenvalue.
	if z.Get(nn-5-1) > tol2*((*sigma)+z.Get(nn-3-1)) && z.Get(nn-2*(*pp)-4-1) > tol2*z.Get(nn-7-1) {
		goto label30
	}

label20:
	;

	z.Set(4*(*n0)-3-1, z.Get(4*(*n0)+(*pp)-3-1)+(*sigma))
	(*n0) = (*n0) - 1
	goto label10

	//     Check  whether E(N0-2) is negligible, 2 eigenvalues.
label30:
	;

	if z.Get(nn-9-1) > tol2*(*sigma) && z.Get(nn-2*(*pp)-8-1) > tol2*z.Get(nn-11-1) {
		goto label50
	}

label40:
	;

	if z.Get(nn-3-1) > z.Get(nn-7-1) {
		s = z.Get(nn - 3 - 1)
		z.Set(nn-3-1, z.Get(nn-7-1))
		z.Set(nn-7-1, s)
	}
	t = half * ((z.Get(nn-7-1) - z.Get(nn-3-1)) + z.Get(nn-5-1))
	if z.Get(nn-5-1) > z.Get(nn-3-1)*tol2 && t != zero {
		s = z.Get(nn-3-1) * (z.Get(nn-5-1) / t)
		if s <= t {
			s = z.Get(nn-3-1) * (z.Get(nn-5-1) / (t * (one + math.Sqrt(one+s/t))))
		} else {
			s = z.Get(nn-3-1) * (z.Get(nn-5-1) / (t + math.Sqrt(t)*math.Sqrt(t+s)))
		}
		t = z.Get(nn-7-1) + (s + z.Get(nn-5-1))
		z.Set(nn-3-1, z.Get(nn-3-1)*(z.Get(nn-7-1)/t))
		z.Set(nn-7-1, t)
	}
	z.Set(4*(*n0)-7-1, z.Get(nn-7-1)+(*sigma))
	z.Set(4*(*n0)-3-1, z.Get(nn-3-1)+(*sigma))
	(*n0) = (*n0) - 2
	goto label10

label50:
	;
	if (*pp) == 2 {
		(*pp) = 0
	}

	//     Reverse the qd-array, if warranted.
	if (*dmin) <= zero || (*n0) < n0in {
		if cbias*z.Get(4*(*i0)+(*pp)-3-1) < z.Get(4*(*n0)+(*pp)-3-1) {
			ipn4 = 4 * ((*i0) + (*n0))
			for j4 = 4 * (*i0); j4 <= 2*((*i0)+(*n0)-1); j4 += 4 {
				temp = z.Get(j4 - 3 - 1)
				z.Set(j4-3-1, z.Get(ipn4-j4-3-1))
				z.Set(ipn4-j4-3-1, temp)
				temp = z.Get(j4 - 2 - 1)
				z.Set(j4-2-1, z.Get(ipn4-j4-2-1))
				z.Set(ipn4-j4-2-1, temp)
				temp = z.Get(j4 - 1 - 1)
				z.Set(j4-1-1, z.Get(ipn4-j4-5-1))
				z.Set(ipn4-j4-5-1, temp)
				temp = z.Get(j4 - 1)
				z.Set(j4-1, z.Get(ipn4-j4-4-1))
				z.Set(ipn4-j4-4-1, temp)
			}
			if (*n0)-(*i0) <= 4 {
				z.Set(4*(*n0)+(*pp)-1-1, z.Get(4*(*i0)+(*pp)-1-1))
				z.Set(4*(*n0)-(*pp)-1, z.Get(4*(*i0)-(*pp)-1))
			}
			(*dmin2) = minf64(*dmin2, z.Get(4*(*n0)+(*pp)-1-1))
			z.Set(4*(*n0)+(*pp)-1-1, minf64(z.Get(4*(*n0)+(*pp)-1-1), z.Get(4*(*i0)+(*pp)-1-1), z.Get(4*(*i0)+(*pp)+3-1)))
			z.Set(4*(*n0)-(*pp)-1, minf64(z.Get(4*(*n0)-(*pp)-1), z.Get(4*(*i0)-(*pp)-1), z.Get(4*(*i0)-(*pp)+4-1)))
			(*qmax) = maxf64(*qmax, z.Get(4*(*i0)+(*pp)-3-1), z.Get(4*(*i0)+(*pp)+1-1))
			(*dmin) = -zero
		}
	}

	//     Choose a shift.
	Dlasq4(i0, n0, z, pp, &n0in, dmin, dmin1, dmin2, dn, dn1, dn2, tau, ttype, g)

	//     Call dqds until DMIN > 0.
label70:
	;

	Dlasq5(i0, n0, z, pp, tau, sigma, dmin, dmin1, dmin2, dn, dn1, dn2, ieee, &eps)

	(*ndiv) = (*ndiv) + ((*n0) - (*i0) + 2)
	(*iter) = (*iter) + 1

	//     Check status.
	if (*dmin) >= zero && (*dmin1) >= zero {
		//        Success.
		goto label90

	} else if (*dmin) < zero && (*dmin1) > zero && z.Get(4*((*n0)-1)-(*pp)-1) < tol*((*sigma)+(*dn1)) && math.Abs(*dn) < tol*(*sigma) {
		//        Convergence hidden by negative DN.
		z.Set(4*((*n0)-1)-(*pp)+2-1, zero)
		(*dmin) = zero
		goto label90
	} else if (*dmin) < zero {
		//        TAU too big. Select new TAU and try again.
		(*nfail) = (*nfail) + 1
		if (*ttype) < -22 {
			//           Failed twice. Play it safe.
			(*tau) = zero
		} else if (*dmin1) > zero {
			//           Late failure. Gives excellent shift.
			(*tau) = ((*tau) + (*dmin)) * (one - two*eps)
			(*ttype) = (*ttype) - 11
		} else {
			//           Early failure. Divide by 4.
			(*tau) = qurtr * (*tau)
			(*ttype) = (*ttype) - 12
		}
		goto label70
	} else if Disnan(int(*dmin)) {
		//        NaN.
		if (*tau) == zero {
			goto label80
		} else {
			(*tau) = zero
			goto label70
		}
	} else {
		//        Possible underflow. Play it safe.
		goto label80
	}

	//     Risk of underflow.
label80:
	;
	Dlasq6(i0, n0, z, pp, dmin, dmin1, dmin2, dn, dn1, dn2)
	(*ndiv) = (*ndiv) + ((*n0) - (*i0) + 2)
	(*iter) = (*iter) + 1
	(*tau) = zero
	//
label90:
	;
	if (*tau) < (*sigma) {
		(*desig) = (*desig) + (*tau)
		t = (*sigma) + (*desig)
		(*desig) = (*desig) - (t - (*sigma))
	} else {
		t = (*sigma) + (*tau)
		(*desig) = (*sigma) - (t - (*tau)) + (*desig)
	}
	(*sigma) = t
}
