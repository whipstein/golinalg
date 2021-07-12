package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
)

// Dget31 tests DLALN2, a routine for solving
//
//    (ca A - w D)X = sB
//
// where A is an NA by NA matrix (NA=1 or 2 only), w is a real (NW=1) or
// complex (NW=2) constant, ca is a real constant, D is an NA by NA real
// diagonal matrix, and B is an NA by NW matrix (when NW=2 the second
// column of B contains the imaginary part of the solution).  The code
// returns X and s, where s is a scale factor, less than or equal to 1,
// which is chosen to avoid overflow in X.
//
// If any singular values of ca A-w D are less than another input
// parameter SMIN, they are perturbed up to SMIN.
//
// The test condition is that the scaled residual
//
//     norm( (ca A-w D)*X - s*B ) /
//           ( math.Max( ulp*norm(ca A-w D), SMIN )*norm(X) )
//
// should be on the order of 1.  Here, ulp is the machine precision.
// Also, it is verified that SCALE is less than or equal to 1, and that
// XNORM = infinity-norm(X).
func Dget31(rmax *float64, lmax *int, ninfo *[]int, knt *int) {
	var bignum, ca, d1, d2, den, eps, four, half, one, res, scale, seven, smin, smlnum, ten, three, tmp, twnone, two, unfl, wi, wr, xnorm, zero float64
	var ia, ib, ica, id1, id2, info, ismin, itrans, iwi, iwr, na, nw int

	ltrans := make([]bool, 2)
	vab := vf(3)
	vca := vf(5)
	vdd := vf(4)
	vsmin := vf(4)
	vwi := vf(4)
	vwr := vf(4)
	a := mf(2, 2, opts)
	b := mf(2, 2, opts)
	x := mf(2, 2, opts)

	zero = 0.0
	half = 0.5
	one = 1.0
	two = 2.0
	three = 3.0
	four = 4.0
	seven = 7.0
	ten = 10.0
	twnone = 21.0
	ltrans[0], ltrans[1] = false, true

	//     Get machine parameters
	eps = golapack.Dlamch(Precision)
	unfl = golapack.Dlamch(Underflow)
	smlnum = golapack.Dlamch(SafeMinimum) / eps
	bignum = one / smlnum
	golapack.Dlabad(&smlnum, &bignum)

	//     Set up test case parameters
	vsmin.Set(0, smlnum)
	vsmin.Set(1, eps)
	vsmin.Set(2, one/(ten*ten))
	vsmin.Set(3, one/eps)
	vab.Set(0, math.Sqrt(smlnum))
	vab.Set(1, one)
	vab.Set(2, math.Sqrt(bignum))
	vwr.Set(0, zero)
	vwr.Set(1, half)
	vwr.Set(2, two)
	vwr.Set(3, one)
	vwi.Set(0, smlnum)
	vwi.Set(1, eps)
	vwi.Set(2, one)
	vwi.Set(3, two)
	vdd.Set(0, math.Sqrt(smlnum))
	vdd.Set(1, one)
	vdd.Set(2, two)
	vdd.Set(3, math.Sqrt(bignum))
	vca.Set(0, zero)
	vca.Set(1, math.Sqrt(smlnum))
	vca.Set(2, eps)
	vca.Set(3, half)
	vca.Set(4, one)

	(*knt) = 0
	(*ninfo)[0] = 0
	(*ninfo)[1] = 0
	(*lmax) = 0
	(*rmax) = zero

	//     Begin test loop
	for id1 = 1; id1 <= 4; id1++ {
		d1 = vdd.Get(id1 - 1)
		for id2 = 1; id2 <= 4; id2++ {
			d2 = vdd.Get(id2 - 1)
			for ica = 1; ica <= 5; ica++ {
				ca = vca.Get(ica - 1)
				for itrans = 0; itrans <= 1; itrans++ {
					for ismin = 1; ismin <= 4; ismin++ {
						smin = vsmin.Get(ismin - 1)

						na = 1
						nw = 1
						for ia = 1; ia <= 3; ia++ {
							a.Set(0, 0, vab.Get(ia-1))
							for ib = 1; ib <= 3; ib++ {
								b.Set(0, 0, vab.Get(ib-1))
								for iwr = 1; iwr <= 4; iwr++ {
									if d1 == one && d2 == one && ca == one {
										wr = vwr.Get(iwr-1) * a.Get(0, 0)
									} else {
										wr = vwr.Get(iwr - 1)
									}
									wi = zero
									golapack.Dlaln2(ltrans[itrans-0], &na, &nw, &smin, &ca, a, func() *int { y := 2; return &y }(), &d1, &d2, b, func() *int { y := 2; return &y }(), &wr, &wi, x, func() *int { y := 2; return &y }(), &scale, &xnorm, &info)
									if info < 0 {
										(*ninfo)[0] = (*ninfo)[0] + 1
									}
									if info > 0 {
										(*ninfo)[1] = (*ninfo)[1] + 1
									}
									res = math.Abs((ca*a.Get(0, 0)-wr*d1)*x.Get(0, 0) - scale*b.Get(0, 0))
									if info == 0 {
										den = math.Max(eps*math.Abs((ca*a.Get(0, 0)-wr*d1)*x.Get(0, 0)), smlnum)
									} else {
										den = math.Max(smin*math.Abs(x.Get(0, 0)), smlnum)
									}
									res = res / den
									if math.Abs(x.Get(0, 0)) < unfl && math.Abs(b.Get(0, 0)) <= smlnum*math.Abs(ca*a.Get(0, 0)-wr*d1) {
										res = zero
									}
									if scale > one {
										res = res + one/eps
									}
									res = res + math.Abs(xnorm-math.Abs(x.Get(0, 0)))/math.Max(smlnum, xnorm)/eps
									if info != 0 && info != 1 {
										res = res + one/eps
									}
									(*knt) = (*knt) + 1
									if res > (*rmax) {
										(*lmax) = (*knt)
										(*rmax) = res
									}
								}
							}
						}

						na = 1
						nw = 2
						for ia = 1; ia <= 3; ia++ {
							a.Set(0, 0, vab.Get(ia-1))
							for ib = 1; ib <= 3; ib++ {
								b.Set(0, 0, vab.Get(ib-1))
								b.Set(0, 1, -half*vab.Get(ib-1))
								for iwr = 1; iwr <= 4; iwr++ {
									if d1 == one && d2 == one && ca == one {
										wr = vwr.Get(iwr-1) * a.Get(0, 0)
									} else {
										wr = vwr.Get(iwr - 1)
									}
									for iwi = 1; iwi <= 4; iwi++ {
										if d1 == one && d2 == one && ca == one {
											wi = vwi.Get(iwi-1) * a.Get(0, 0)
										} else {
											wi = vwi.Get(iwi - 1)
										}
										golapack.Dlaln2(ltrans[itrans-0], &na, &nw, &smin, &ca, a, func() *int { y := 2; return &y }(), &d1, &d2, b, func() *int { y := 2; return &y }(), &wr, &wi, x, func() *int { y := 2; return &y }(), &scale, &xnorm, &info)
										if info < 0 {
											(*ninfo)[0] = (*ninfo)[0] + 1
										}
										if info > 0 {
											(*ninfo)[1] = (*ninfo)[1] + 1
										}
										res = math.Abs((ca*a.Get(0, 0)-wr*d1)*x.Get(0, 0) + (wi*d1)*x.Get(0, 1) - scale*b.Get(0, 0))
										res = res + math.Abs((-wi*d1)*x.Get(0, 0)+(ca*a.Get(0, 0)-wr*d1)*x.Get(0, 1)-scale*b.Get(0, 1))
										if info == 0 {
											den = math.Max(eps*(math.Max(math.Abs(ca*a.Get(0, 0)-wr*d1), math.Abs(d1*wi))*(math.Abs(x.Get(0, 0))+math.Abs(x.Get(0, 1)))), smlnum)
										} else {
											den = math.Max(smin*(math.Abs(x.Get(0, 0))+math.Abs(x.Get(0, 1))), smlnum)
										}
										res = res / den
										if math.Abs(x.Get(0, 0)) < unfl && math.Abs(x.Get(0, 1)) < unfl && math.Abs(b.Get(0, 0)) <= smlnum*math.Abs(ca*a.Get(0, 0)-wr*d1) {
											res = zero
										}
										if scale > one {
											res = res + one/eps
										}
										res = res + math.Abs(xnorm-math.Abs(x.Get(0, 0))-math.Abs(x.Get(0, 1)))/math.Max(smlnum, xnorm)/eps
										if info != 0 && info != 1 {
											res = res + one/eps
										}
										(*knt) = (*knt) + 1
										if res > (*rmax) {
											(*lmax) = (*knt)
											(*rmax) = res
										}
									}
								}
							}
						}

						na = 2
						nw = 1
						for ia = 1; ia <= 3; ia++ {
							a.Set(0, 0, vab.Get(ia-1))
							a.Set(0, 1, -three*vab.Get(ia-1))
							a.Set(1, 0, -seven*vab.Get(ia-1))
							a.Set(1, 1, twnone*vab.Get(ia-1))
							for ib = 1; ib <= 3; ib++ {
								b.Set(0, 0, vab.Get(ib-1))
								b.Set(1, 0, -two*vab.Get(ib-1))
								for iwr = 1; iwr <= 4; iwr++ {
									if d1 == one && d2 == one && ca == one {
										wr = vwr.Get(iwr-1) * a.Get(0, 0)
									} else {
										wr = vwr.Get(iwr - 1)
									}
									wi = zero
									golapack.Dlaln2(ltrans[itrans-0], &na, &nw, &smin, &ca, a, func() *int { y := 2; return &y }(), &d1, &d2, b, func() *int { y := 2; return &y }(), &wr, &wi, x, func() *int { y := 2; return &y }(), &scale, &xnorm, &info)
									if info < 0 {
										(*ninfo)[0] = (*ninfo)[0] + 1
									}
									if info > 0 {
										(*ninfo)[1] = (*ninfo)[1] + 1
									}
									if itrans == 1 {
										tmp = a.Get(0, 1)
										a.Set(0, 1, a.Get(1, 0))
										a.Set(1, 0, tmp)
									}
									res = math.Abs((ca*a.Get(0, 0)-wr*d1)*x.Get(0, 0) + (ca*a.Get(0, 1))*x.Get(1, 0) - scale*b.Get(0, 0))
									res = res + math.Abs((ca*a.Get(1, 0))*x.Get(0, 0)+(ca*a.Get(1, 1)-wr*d2)*x.Get(1, 0)-scale*b.Get(1, 0))
									if info == 0 {
										den = math.Max(eps*(math.Max(math.Abs(ca*a.Get(0, 0)-wr*d1)+math.Abs(ca*a.Get(0, 1)), math.Abs(ca*a.Get(1, 0))+math.Abs(ca*a.Get(1, 1)-wr*d2))*math.Max(math.Abs(x.Get(0, 0)), math.Abs(x.Get(1, 0)))), smlnum)
									} else {
										den = math.Max(eps*(math.Max(smin/eps, math.Max(math.Abs(ca*a.Get(0, 0)-wr*d1)+math.Abs(ca*a.Get(0, 1)), math.Abs(ca*a.Get(1, 0))+math.Abs(ca*a.Get(1, 1)-wr*d2)))*math.Max(math.Abs(x.Get(0, 0)), math.Abs(x.Get(1, 0)))), smlnum)
									}
									res = res / den
									if math.Abs(x.Get(0, 0)) < unfl && math.Abs(x.Get(1, 0)) < unfl && math.Abs(b.Get(0, 0))+math.Abs(b.Get(1, 0)) <= smlnum*(math.Abs(ca*a.Get(0, 0)-wr*d1)+math.Abs(ca*a.Get(0, 1))+math.Abs(ca*a.Get(1, 0))+math.Abs(ca*a.Get(1, 1)-wr*d2)) {
										res = zero
									}
									if scale > one {
										res = res + one/eps
									}
									res = res + math.Abs(xnorm-math.Max(math.Abs(x.Get(0, 0)), math.Abs(x.Get(1, 0))))/math.Max(smlnum, xnorm)/eps
									if info != 0 && info != 1 {
										res = res + one/eps
									}
									(*knt) = (*knt) + 1
									if res > (*rmax) {
										(*lmax) = (*knt)
										(*rmax) = res
									}
								}
							}
						}

						na = 2
						nw = 2
						for ia = 1; ia <= 3; ia++ {
							a.Set(0, 0, vab.Get(ia-1)*two)
							a.Set(0, 1, -three*vab.Get(ia-1))
							a.Set(1, 0, -seven*vab.Get(ia-1))
							a.Set(1, 1, twnone*vab.Get(ia-1))
							for ib = 1; ib <= 3; ib++ {
								b.Set(0, 0, vab.Get(ib-1))
								b.Set(1, 0, -two*vab.Get(ib-1))
								b.Set(0, 1, four*vab.Get(ib-1))
								b.Set(1, 1, -seven*vab.Get(ib-1))
								for iwr = 1; iwr <= 4; iwr++ {
									if d1 == one && d2 == one && ca == one {
										wr = vwr.Get(iwr-1) * a.Get(0, 0)
									} else {
										wr = vwr.Get(iwr - 1)
									}
									for iwi = 1; iwi <= 4; iwi++ {
										if d1 == one && d2 == one && ca == one {
											wi = vwi.Get(iwi-1) * a.Get(0, 0)
										} else {
											wi = vwi.Get(iwi - 1)
										}
										golapack.Dlaln2(ltrans[itrans-0], &na, &nw, &smin, &ca, a, func() *int { y := 2; return &y }(), &d1, &d2, b, func() *int { y := 2; return &y }(), &wr, &wi, x, func() *int { y := 2; return &y }(), &scale, &xnorm, &info)
										if info < 0 {
											(*ninfo)[0] = (*ninfo)[0] + 1
										}
										if info > 0 {
											(*ninfo)[1] = (*ninfo)[1] + 1
										}
										if itrans == 1 {
											tmp = a.Get(0, 1)
											a.Set(0, 1, a.Get(1, 0))
											a.Set(1, 0, tmp)
										}
										res = math.Abs((ca*a.Get(0, 0)-wr*d1)*x.Get(0, 0) + (ca*a.Get(0, 1))*x.Get(1, 0) + (wi*d1)*x.Get(0, 1) - scale*b.Get(0, 0))
										res = res + math.Abs((ca*a.Get(0, 0)-wr*d1)*x.Get(0, 1)+(ca*a.Get(0, 1))*x.Get(1, 1)-(wi*d1)*x.Get(0, 0)-scale*b.Get(0, 1))
										res = res + math.Abs((ca*a.Get(1, 0))*x.Get(0, 0)+(ca*a.Get(1, 1)-wr*d2)*x.Get(1, 0)+(wi*d2)*x.Get(1, 1)-scale*b.Get(1, 0))
										res = res + math.Abs((ca*a.Get(1, 0))*x.Get(0, 1)+(ca*a.Get(1, 1)-wr*d2)*x.Get(1, 1)-(wi*d2)*x.Get(1, 0)-scale*b.Get(1, 1))
										if info == 0 {
											den = math.Max(eps*(math.Max(math.Abs(ca*a.Get(0, 0)-wr*d1)+math.Abs(ca*a.Get(0, 1))+math.Abs(wi*d1), math.Abs(ca*a.Get(1, 0))+math.Abs(ca*a.Get(1, 1)-wr*d2)+math.Abs(wi*d2))*math.Max(math.Abs(x.Get(0, 0))+math.Abs(x.Get(1, 0)), math.Abs(x.Get(0, 1))+math.Abs(x.Get(1, 1)))), smlnum)
										} else {
											den = math.Max(eps*(math.Max(smin/eps, math.Max(math.Abs(ca*a.Get(0, 0)-wr*d1)+math.Abs(ca*a.Get(0, 1))+math.Abs(wi*d1), math.Abs(ca*a.Get(1, 0))+math.Abs(ca*a.Get(1, 1)-wr*d2)+math.Abs(wi*d2)))*math.Max(math.Abs(x.Get(0, 0))+math.Abs(x.Get(1, 0)), math.Abs(x.Get(0, 1))+math.Abs(x.Get(1, 1)))), smlnum)
										}
										res = res / den
										if math.Abs(x.Get(0, 0)) < unfl && math.Abs(x.Get(1, 0)) < unfl && math.Abs(x.Get(0, 1)) < unfl && math.Abs(x.Get(1, 1)) < unfl && math.Abs(b.Get(0, 0))+math.Abs(b.Get(1, 0)) <= smlnum*(math.Abs(ca*a.Get(0, 0)-wr*d1)+math.Abs(ca*a.Get(0, 1))+math.Abs(ca*a.Get(1, 0))+math.Abs(ca*a.Get(1, 1)-wr*d2)+math.Abs(wi*d2)+math.Abs(wi*d1)) {
											res = zero
										}
										if scale > one {
											res = res + one/eps
										}
										res = res + math.Abs(xnorm-math.Max(math.Abs(x.Get(0, 0))+math.Abs(x.Get(0, 1)), math.Abs(x.Get(1, 0))+math.Abs(x.Get(1, 1))))/math.Max(smlnum, xnorm)/eps
										if info != 0 && info != 1 {
											res = res + one/eps
										}
										(*knt) = (*knt) + 1
										if res > (*rmax) {
											(*lmax) = (*knt)
											(*rmax) = res
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}
