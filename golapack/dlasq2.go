package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dlasq2 computes all the eigenvalues of the symmetric positive
// definite tridiagonal matrix associated with the qd array Z to high
// relative accuracy are computed to high relative accuracy, in the
// absence of denormalization, underflow and overflow.
//
// To see the relation of Z to the tridiagonal matrix, let L be a
// unit lower bidiagonal matrix with subdiagonals Z(2,4,6,,..) and
// let U be an upper bidiagonal matrix with 1's above and diagonal
// Z(1,3,5,,..). The tridiagonal is L*U or, if you prefer, the
// symmetric tridiagonal to which it is similar.
//
// Note : DLASQ2 defines a logical variable, IEEE, which is true
// on machines which follow ieee-754 floating-point standard in their
// handling of infinities and NaNs, and false otherwise. This variable
// is passed to DLASQ3.
func Dlasq2(n *int, z *mat.Vector, info *int) {
	var ieee bool
	var cbias, d, dee, deemin, desig, dmin, dmin1, dmin2, dn, dn1, dn2, e, emax, emin, eps, four, g, half, hundrd, oldemn, one, qmax, qmin, s, safmin, sigma, t, tau, temp, tempe, tempq, tol, tol2, trace, two, zero, zmax float64
	var i0, i1, i4, iinfo, ipn4, iter, iwhila, iwhilb, k, kmin, n0, n1, nbig, ndiv, nfail, pp, splt, ttype int

	cbias = 1.50
	zero = 0.0
	half = 0.5
	one = 1.0
	two = 2.0
	four = 4.0
	hundrd = 100.0

	//     Test the input arguments.
	//     (in case DLASQ2 is not called by DLASQ1)
	(*info) = 0
	eps = Dlamch(Precision)
	safmin = Dlamch(SafeMinimum)
	tol = eps * hundrd
	tol2 = math.Pow(tol, 2)

	if (*n) < 0 {
		(*info) = -1
		gltest.Xerbla([]byte("DLASQ2"), 1)
		return
	} else if (*n) == 0 {
		return
	} else if (*n) == 1 {
		//        1-by-1 case.
		if z.Get(0) < zero {
			(*info) = -201
			gltest.Xerbla([]byte("DLASQ2"), 2)
		}
		return
	} else if (*n) == 2 {
		//        2-by-2 case.
		if z.Get(1) < zero || z.Get(2) < zero {
			(*info) = -2
			gltest.Xerbla([]byte("DLASQ2"), 2)
			return
		} else if z.Get(2) > z.Get(0) {
			d = z.Get(2)
			z.Set(2, z.Get(0))
			z.Set(0, d)
		}
		z.Set(4, z.Get(0)+z.Get(1)+z.Get(2))
		if z.Get(1) > z.Get(2)*tol2 {
			t = half * ((z.Get(0) - z.Get(2)) + z.Get(1))
			s = z.Get(2) * (z.Get(1) / t)
			if s <= t {
				s = z.Get(2) * (z.Get(1) / (t * (one + math.Sqrt(one+s/t))))
			} else {
				s = z.Get(2) * (z.Get(1) / (t + math.Sqrt(t)*math.Sqrt(t+s)))
			}
			t = z.Get(0) + (s + z.Get(1))
			z.Set(2, z.Get(2)*(z.Get(0)/t))
			z.Set(0, t)
		}
		z.Set(1, z.Get(2))
		z.Set(5, z.Get(1)+z.Get(0))
		return
	}

	//     Check for negative data and compute sums of q's and e's.
	z.Set(2*(*n)-1, zero)
	emin = z.Get(1)
	qmax = zero
	zmax = zero
	d = zero
	e = zero

	for k = 1; k <= 2*((*n)-1); k += 2 {
		if z.Get(k-1) < zero {
			(*info) = -(200 + k)
			gltest.Xerbla([]byte("DLASQ2"), 2)
			return
		} else if z.Get(k+1-1) < zero {
			(*info) = -(200 + k + 1)
			gltest.Xerbla([]byte("DLASQ2"), 2)
			return
		}
		d = d + z.Get(k-1)
		e = e + z.Get(k+1-1)
		qmax = maxf64(qmax, z.Get(k-1))
		emin = minf64(emin, z.Get(k+1-1))
		zmax = maxf64(qmax, zmax, z.Get(k+1-1))
		//Label10:
	}
	if z.Get(2*(*n)-1-1) < zero {
		(*info) = -(200 + 2*(*n) - 1)
		gltest.Xerbla([]byte("DLASQ2"), 2)
		return
	}
	d = d + z.Get(2*(*n)-1-1)
	qmax = maxf64(qmax, z.Get(2*(*n)-1-1))
	zmax = maxf64(qmax, zmax)

	//     Check for diagonality.
	if e == zero {
		for k = 2; k <= (*n); k++ {
			z.Set(k-1, z.Get(2*k-1-1))
		}
		Dlasrt('D', n, z, &iinfo)
		z.Set(2*(*n)-1-1, d)
		return
	}

	trace = d + e

	//     Check for zero data.
	if trace == zero {
		z.Set(2*(*n)-1-1, zero)
		return
	}

	//     Check whether the machine is IEEE conformable.
	ieee = Ilaenv(func() *int { y := 10; return &y }(), []byte("DLASQ2"), []byte("N"), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 3; return &y }(), func() *int { y := 4; return &y }()) == 1 && Ilaenv(func() *int { y := 11; return &y }(), []byte("DLASQ2"), []byte("N"), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 3; return &y }(), func() *int { y := 4; return &y }()) == 1

	//     Rearrange data for locality: Z=(q1,qq1,e1,ee1,q2,qq2,e2,ee2,...).
	for k = 2 * (*n); k >= 2; k -= 2 {
		z.Set(2*k-1, zero)
		z.Set(2*k-1-1, z.Get(k-1))
		z.Set(2*k-2-1, zero)
		z.Set(2*k-3-1, z.Get(k-1-1))
	}

	i0 = 1
	n0 = (*n)

	//     Reverse the qd-array, if warranted.
	if cbias*z.Get(4*i0-3-1) < z.Get(4*n0-3-1) {
		ipn4 = 4 * (i0 + n0)
		for i4 = 4 * i0; i4 <= 2*(i0+n0-1); i4 += 4 {
			temp = z.Get(i4 - 3 - 1)
			z.Set(i4-3-1, z.Get(ipn4-i4-3-1))
			z.Set(ipn4-i4-3-1, temp)
			temp = z.Get(i4 - 1 - 1)
			z.Set(i4-1-1, z.Get(ipn4-i4-5-1))
			z.Set(ipn4-i4-5-1, temp)
		}
	}

	//     Initial split checking via dqd and Li's test.
	pp = 0

	for k = 1; k <= 2; k++ {

		d = z.Get(4*n0 + pp - 3 - 1)
		for i4 = 4*(n0-1) + pp; i4 >= 4*i0+pp; i4 -= 4 {
			if z.Get(i4-1-1) <= tol2*d {
				z.Set(i4-1-1, -zero)
				d = z.Get(i4 - 3 - 1)
			} else {
				d = z.Get(i4-3-1) * (d / (d + z.Get(i4-1-1)))
			}
		}

		//        dqd maps Z to ZZ plus Li's test.
		emin = z.Get(4*i0 + pp + 1 - 1)
		d = z.Get(4*i0 + pp - 3 - 1)
		for i4 = 4*i0 + pp; i4 <= 4*(n0-1)+pp; i4 += 4 {
			z.Set(i4-2*pp-2-1, d+z.Get(i4-1-1))
			if z.Get(i4-1-1) <= tol2*d {
				z.Set(i4-1-1, -zero)
				z.Set(i4-2*pp-2-1, d)
				z.Set(i4-2*pp-1, zero)
				d = z.Get(i4 + 1 - 1)
			} else if safmin*z.Get(i4+1-1) < z.Get(i4-2*pp-2-1) && safmin*z.Get(i4-2*pp-2-1) < z.Get(i4+1-1) {
				temp = z.Get(i4+1-1) / z.Get(i4-2*pp-2-1)
				z.Set(i4-2*pp-1, z.Get(i4-1-1)*temp)
				d = d * temp
			} else {
				z.Set(i4-2*pp-1, z.Get(i4+1-1)*(z.Get(i4-1-1)/z.Get(i4-2*pp-2-1)))
				d = z.Get(i4+1-1) * (d / z.Get(i4-2*pp-2-1))
			}
			emin = minf64(emin, z.Get(i4-2*pp-1))
		}
		z.Set(4*n0-pp-2-1, d)

		//        Now find qmax.
		qmax = z.Get(4*i0 - pp - 2 - 1)
		for i4 = 4*i0 - pp + 2; i4 <= 4*n0-pp-2; i4 += 4 {
			qmax = maxf64(qmax, z.Get(i4-1))
		}

		//        Prepare for the next iteration on K.
		pp = 1 - pp
	}

	//     Initialise variables to pass to DLASQ3.
	ttype = 0
	dmin1 = zero
	dmin2 = zero
	dn = zero
	dn1 = zero
	dn2 = zero
	g = zero
	tau = zero

	iter = 2
	nfail = 0
	ndiv = 2 * (n0 - i0)

	for iwhila = 1; iwhila <= (*n)+1; iwhila++ {
		if n0 < 1 {
			goto label170
		}

		//        While array unfinished do
		//
		//        E(N0) holds the value of SIGMA when submatrix in I0:N0
		//        splits from the rest of the array, but is negated.
		desig = zero
		if n0 == (*n) {
			sigma = zero
		} else {
			sigma = -z.Get(4*n0 - 1 - 1)
		}
		if sigma < zero {
			(*info) = 1
			return
		}

		//        Find last unreduced submatrix's top index I0, find QMAX and
		//        EMIN. Find Gershgorin-type bound if Q's much greater than E's.
		emax = zero
		if n0 > i0 {
			emin = math.Abs(z.Get(4*n0 - 5 - 1))
		} else {
			emin = zero
		}
		qmin = z.Get(4*n0 - 3 - 1)
		qmax = qmin
		for i4 = 4 * n0; i4 >= 8; i4 -= 4 {
			if z.Get(i4-5-1) <= zero {
				goto label100
			}
			if qmin >= four*emax {
				qmin = minf64(qmin, z.Get(i4-3-1))
				emax = maxf64(emax, z.Get(i4-5-1))
			}
			qmax = maxf64(qmax, z.Get(i4-7-1)+z.Get(i4-5-1))
			emin = minf64(emin, z.Get(i4-5-1))
		}
		i4 = 4

	label100:
		;
		i0 = i4 / 4
		pp = 0

		if n0-i0 > 1 {
			dee = z.Get(4*i0 - 3 - 1)
			deemin = dee
			kmin = i0
			for i4 = 4*i0 + 1; i4 <= 4*n0-3; i4 += 4 {
				dee = z.Get(i4-1) * (dee / (dee + z.Get(i4-2-1)))
				if dee <= deemin {
					deemin = dee
					kmin = (i4 + 3) / 4
				}
			}
			if (kmin-i0)*2 < n0-kmin && deemin <= half*z.Get(4*n0-3-1) {
				ipn4 = 4 * (i0 + n0)
				pp = 2
				for i4 = 4 * i0; i4 <= 2*(i0+n0-1); i4 += 4 {
					temp = z.Get(i4 - 3 - 1)
					z.Set(i4-3-1, z.Get(ipn4-i4-3-1))
					z.Set(ipn4-i4-3-1, temp)
					temp = z.Get(i4 - 2 - 1)
					z.Set(i4-2-1, z.Get(ipn4-i4-2-1))
					z.Set(ipn4-i4-2-1, temp)
					temp = z.Get(i4 - 1 - 1)
					z.Set(i4-1-1, z.Get(ipn4-i4-5-1))
					z.Set(ipn4-i4-5-1, temp)
					temp = z.Get(i4 - 1)
					z.Set(i4-1, z.Get(ipn4-i4-4-1))
					z.Set(ipn4-i4-4-1, temp)
				}
			}
		}

		//        Put -(initial shift) into DMIN.
		dmin = -maxf64(zero, qmin-two*math.Sqrt(qmin)*math.Sqrt(emax))

		//        Now I0:N0 is unreduced.
		//        PP = 0 for ping, PP = 1 for pong.
		//        PP = 2 indicates that flipping was applied to the Z array and
		//               and that the tests for deflation upon entry in DLASQ3
		//               should not be performed.
		nbig = 100 * (n0 - i0 + 1)
		for iwhilb = 1; iwhilb <= nbig; iwhilb++ {
			if i0 > n0 {
				goto label150
			}

			//           While submatrix unfinished take a good dqds step.
			Dlasq3(&i0, &n0, z, &pp, &dmin, &sigma, &desig, &qmax, &nfail, &iter, &ndiv, &ieee, &ttype, &dmin1, &dmin2, &dn, &dn1, &dn2, &g, &tau)

			pp = 1 - pp

			//           When EMIN is very small check for splits.
			if pp == 0 && n0-i0 >= 3 {
				if z.Get(4*n0-1) <= tol2*qmax || z.Get(4*n0-1-1) <= tol2*sigma {
					splt = i0 - 1
					qmax = z.Get(4*i0 - 3 - 1)
					emin = z.Get(4*i0 - 1 - 1)
					oldemn = z.Get(4*i0 - 1)
					for i4 = 4 * i0; i4 <= 4*(n0-3); i4 += 4 {
						if z.Get(i4-1) <= tol2*z.Get(i4-3-1) || z.Get(i4-1-1) <= tol2*sigma {
							z.Set(i4-1-1, -sigma)
							splt = i4 / 4
							qmax = zero
							emin = z.Get(i4 + 3 - 1)
							oldemn = z.Get(i4 + 4 - 1)
						} else {
							qmax = maxf64(qmax, z.Get(i4+1-1))
							emin = minf64(emin, z.Get(i4-1-1))
							oldemn = minf64(oldemn, z.Get(i4-1))
						}
					}
					z.Set(4*n0-1-1, emin)
					z.Set(4*n0-1, oldemn)
					i0 = splt + 1
				}
			}

		}

		(*info) = 2

		//        Maximum number of iterations exceeded, restore the shift
		//        SIGMA and place the new d's and e's in a qd array.
		//        This might need to be done for several blocks
		i1 = i0
		n1 = n0
	label145:
		;
		tempq = z.Get(4*i0 - 3 - 1)
		z.Set(4*i0-3-1, z.Get(4*i0-3-1)+sigma)
		for k = i0 + 1; k <= n0; k++ {
			tempe = z.Get(4*k - 5 - 1)
			z.Set(4*k-5-1, z.Get(4*k-5-1)*(tempq/z.Get(4*k-7-1)))
			tempq = z.Get(4*k - 3 - 1)
			z.Set(4*k-3-1, z.Get(4*k-3-1)+sigma+tempe-z.Get(4*k-5-1))
		}

		//        Prepare to do this on the previous block if there is one
		if i1 > 1 {
			n1 = i1 - 1
			for (i1 >= 2) && (z.Get(4*i1-5-1) >= zero) {
				i1 = i1 - 1
			}
			sigma = -z.Get(4*n1 - 1 - 1)
			goto label145
		}
		for k = 1; k <= (*n); k++ {
			z.Set(2*k-1-1, z.Get(4*k-3-1))

			//        Only the block 1..N0 is unfinished.  The rest of the e's
			//        must be essentially zero, although sometimes other data
			//        has been stored in them.
			if k < n0 {
				z.Set(2*k-1, z.Get(4*k-1-1))
			} else {
				z.Set(2*k-1, 0)
			}
		}
		return

	label150:
	}

	(*info) = 3
	return

label170:
	;

	//     Move q's to the front.
	for k = 2; k <= (*n); k++ {
		z.Set(k-1, z.Get(4*k-3-1))
	}

	//     Sort and compute sum of eigenvalues.
	Dlasrt('D', n, z, &iinfo)

	e = zero
	for k = (*n); k >= 1; k-- {
		e = e + z.Get(k-1)
	}

	//     Store trace, sum(eigenvalues) and information on performance.
	z.Set(2*(*n)+1-1, trace)
	z.Set(2*(*n)+2-1, e)
	z.Set(2*(*n)+3-1, float64(iter))
	z.Set(2*(*n)+4-1, float64(ndiv)/float64(math.Pow(float64(*n), 2)))
	z.Set(2*(*n)+5-1, hundrd*float64(nfail)/float64(iter))
}
