package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlasq3 checks for deflation, computes a shift (TAU) and calls dqds.
// In case of failure it changes shifts, and tries again until output
// is positive.
func Dlasq3(i0, n0 int, z *mat.Vector, pp int, dmin, sigma, desig, qmax float64, nfail, iter, ndiv int, ieee bool, ttype int, dmin1, dmin2, dn, dn1, dn2, g, tau float64) (n0Out, ppOut int, dminOut, sigmaOut, desigOut, qmaxOut float64, nfailOut, iterOut, ndivOut, ttypeOut int, dmin1Out, dmin2Out, dnOut, dn1Out, dn2Out, gOut, tauOut float64) {
	var cbias, eps, half, hundrd, one, qurtr, s, t, temp, tol, tol2, two, zero float64
	var ipn4, j4, n0in, nn int

	cbias = 1.50
	zero = 0.0
	qurtr = 0.250
	half = 0.5
	one = 1.0
	two = 2.0
	hundrd = 100.0
	n0Out = n0
	ppOut = pp
	dminOut = dmin
	sigmaOut = sigma
	desigOut = desig
	qmaxOut = qmax
	nfailOut = nfail
	iterOut = iter
	ndivOut = ndiv
	ttypeOut = ttype
	dmin1Out = dmin1
	dmin2Out = dmin2
	dnOut = dn
	dn1Out = dn1
	dn2Out = dn2
	gOut = g
	tauOut = tau

	n0in = n0
	eps = Dlamch(Precision)
	tol = eps * hundrd
	tol2 = math.Pow(tol, 2)

	//     Check for deflation.
label10:
	;

	if n0Out < i0 {
		return
	}
	if n0Out == i0 {
		goto label20
	}
	nn = 4*n0Out + ppOut
	if n0Out == (i0 + 1) {
		goto label40
	}

	//     Check whether E(N0-1) is negligible, 1 eigenvalue.
	if z.Get(nn-5-1) > tol2*(sigmaOut+z.Get(nn-3-1)) && z.Get(nn-2*ppOut-4-1) > tol2*z.Get(nn-7-1) {
		goto label30
	}

label20:
	;

	z.Set(4*n0Out-3-1, z.Get(4*n0Out+ppOut-3-1)+sigmaOut)
	n0Out = n0Out - 1
	goto label10

	//     Check  whether E(N0-2) is negligible, 2 eigenvalues.
label30:
	;

	if z.Get(nn-9-1) > tol2*sigmaOut && z.Get(nn-2*ppOut-8-1) > tol2*z.Get(nn-11-1) {
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
	z.Set(4*n0Out-7-1, z.Get(nn-7-1)+sigmaOut)
	z.Set(4*n0Out-3-1, z.Get(nn-3-1)+sigmaOut)
	n0Out = n0Out - 2
	goto label10

label50:
	;
	if ppOut == 2 {
		ppOut = 0
	}

	//     Reverse the qd-array, if warranted.
	if dminOut <= zero || n0Out < n0in {
		if cbias*z.Get(4*i0+ppOut-3-1) < z.Get(4*n0Out+ppOut-3-1) {
			ipn4 = 4 * (i0 + n0Out)
			for j4 = 4 * i0; j4 <= 2*(i0+n0Out-1); j4 += 4 {
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
			if n0Out-i0 <= 4 {
				z.Set(4*n0Out+ppOut-1-1, z.Get(4*i0+ppOut-1-1))
				z.Set(4*n0Out-ppOut-1, z.Get(4*i0-ppOut-1))
			}
			dmin2Out = math.Min(dmin2Out, z.Get(4*n0Out+ppOut-1-1))
			z.Set(4*n0Out+ppOut-1-1, math.Min(z.Get(4*n0Out+ppOut-1-1), math.Min(z.Get(4*i0+ppOut-1-1), z.Get(4*i0+ppOut+3-1))))
			z.Set(4*n0Out-ppOut-1, math.Min(z.Get(4*n0Out-ppOut-1), math.Min(z.Get(4*i0-ppOut-1), z.Get(4*i0-ppOut+4-1))))
			qmaxOut = math.Max(qmaxOut, math.Max(z.Get(4*i0+ppOut-3-1), z.Get(4*i0+ppOut)))
			dminOut = -zero
		}
	}

	//     Choose a shift.
	tauOut, ttypeOut, gOut = Dlasq4(i0, n0Out, z, ppOut, n0in, dminOut, dmin1Out, dmin2Out, dnOut, dn1Out, dn2Out, gOut)

	//     Call dqds until DMIN > 0.
label70:
	;

	dminOut, dmin1Out, dmin2Out, dnOut, dn1Out, dn2Out = Dlasq5(i0, n0Out, z, ppOut, tauOut, sigmaOut, ieee, eps)

	ndivOut = ndivOut + (n0Out - i0 + 2)
	iterOut = iterOut + 1

	//     Check status.
	if dminOut >= zero && dmin1Out >= zero {
		//        Success.
		goto label90

	} else if dminOut < zero && dmin1Out > zero && z.Get(4*(n0Out-1)-ppOut-1) < tol*(sigmaOut+dn1Out) && math.Abs(dnOut) < tol*sigmaOut {
		//        Convergence hidden by negative DN.
		z.Set(4*(n0Out-1)-ppOut+2-1, zero)
		dminOut = zero
		goto label90
	} else if dminOut < zero {
		//        TAU too big. Select new TAU and try again.
		nfailOut = nfailOut + 1
		if ttypeOut < -22 {
			//           Failed twice. Play it safe.
			tauOut = zero
		} else if dmin1Out > zero {
			//           Late failure. Gives excellent shift.
			tauOut = (tauOut + dminOut) * (one - two*eps)
			ttypeOut = ttypeOut - 11
		} else {
			//           Early failure. Divide by 4.
			tauOut = qurtr * tauOut
			ttypeOut = ttypeOut - 12
		}
		goto label70
	} else if Disnan(int(dminOut)) {
		//        NaN.
		if tauOut == zero {
			goto label80
		} else {
			tauOut = zero
			goto label70
		}
	} else {
		//        Possible underflow. Play it safe.
		goto label80
	}

	//     Risk of underflow.
label80:
	;
	dminOut, dmin1Out, dmin2Out, dnOut, dn1Out, dn2Out = Dlasq6(i0, n0Out, z, ppOut)
	ndivOut = ndivOut + (n0Out - i0 + 2)
	iterOut = iterOut + 1
	tauOut = zero
	//
label90:
	;
	if tauOut < sigmaOut {
		desigOut = desigOut + tauOut
		t = sigmaOut + desigOut
		desigOut = desigOut - (t - sigmaOut)
	} else {
		t = sigmaOut + tauOut
		desigOut = sigmaOut - (t - tauOut) + desigOut
	}
	sigmaOut = t

	return
}
