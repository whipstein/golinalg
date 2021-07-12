package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
)

// Dget32 tests DLASY2, a routine for solving
//
//         op(TL)*X + ISGN*X*op(TR) = SCALE*B
//
// where TL is N1 by N1, TR is N2 by N2, and N1,N2 =1 or 2 only.
// X and B are N1 by N2, op() is an optional transpose, an
// ISGN = 1 or -1. SCALE is chosen less than or equal to 1 to
// avoid overflow in X.
//
// The test condition is that the scaled residual
//
// norm( op(TL)*X + ISGN*X*op(TR) = SCALE*B )
//      / ( math.Max( ulp*norm(TL), ulp*norm(TR)) * norm(X), SMLNUM )
//
// should be on the order of 1. Here, ulp is the machine precision.
// Also, it is verified that SCALE is less than or equal to 1, and
// that XNORM = infinity-norm(X).
func Dget32(rmax *float64, lmax, ninfo, knt *int) {
	var ltranl, ltranr bool
	var bignum, den, eight, eps, four, one, res, scale, sgn, smlnum, tmp, tnrm, two, xnorm, xnrm, zero float64
	var ib, ib1, ib2, ib3, info, isgn, itl, itlscl, itr, itranl, itranr, itrscl, n1, n2 int

	val := vf(3)
	b := mf(2, 2, opts)
	tl := mf(2, 2, opts)
	tr := mf(2, 2, opts)
	x := mf(2, 2, opts)

	itval := make([]int, 2*2*8)

	zero = 0.0
	one = 1.0
	two = 2.0
	four = 4.0
	eight = 8.0

	itval[0+(0+(0)*(2))*2], itval[1+(0+(0)*(2))*2], itval[0+(1+(0)*(2))*2], itval[1+(1+(0)*(2))*2], itval[0+(0+(1)*(2))*2], itval[1+(0+(1)*(2))*2], itval[0+(1+(1)*(2))*2], itval[1+(1+(1)*(2))*2], itval[0+(0+(2)*(2))*2], itval[1+(0+(2)*(2))*2], itval[0+(1+(2)*(2))*2], itval[1+(1+(2)*(2))*2], itval[0+(0+(3)*(2))*2], itval[1+(0+(3)*(2))*2], itval[0+(1+(3)*(2))*2], itval[1+(1+(3)*(2))*2], itval[0+(0+(4)*(2))*2], itval[1+(0+(4)*(2))*2], itval[0+(1+(4)*(2))*2], itval[1+(1+(4)*(2))*2], itval[0+(0+(5)*(2))*2], itval[1+(0+(5)*(2))*2], itval[0+(1+(5)*(2))*2], itval[1+(1+(5)*(2))*2], itval[0+(0+(6)*(2))*2], itval[1+(0+(6)*(2))*2], itval[0+(1+(6)*(2))*2], itval[1+(1+(6)*(2))*2], itval[0+(0+(7)*(2))*2], itval[1+(0+(7)*(2))*2], itval[0+(1+(7)*(2))*2], itval[1+(1+(7)*(2))*2] = 8, 4, 2, 1, 4, 8, 1, 2, 2, 1, 8, 4, 1, 2, 4, 8, 9, 4, 2, 1, 4, 9, 1, 2, 2, 1, 9, 4, 1, 2, 4, 9

	//     Get machine parameters
	eps = golapack.Dlamch(Precision)
	smlnum = golapack.Dlamch(SafeMinimum) / eps
	bignum = one / smlnum
	golapack.Dlabad(&smlnum, &bignum)

	//     Set up test case parameters
	val.Set(0, math.Sqrt(smlnum))
	val.Set(1, one)
	val.Set(2, math.Sqrt(bignum))

	(*knt) = 0
	(*ninfo) = 0
	(*lmax) = 0
	(*rmax) = zero

	//     Begin test loop
	for itranl = 0; itranl <= 1; itranl++ {
		for itranr = 0; itranr <= 1; itranr++ {
			for isgn = -1; isgn <= 1; isgn += 2 {
				sgn = float64(isgn)
				ltranl = itranl == 1
				ltranr = itranr == 1

				n1 = 1
				n2 = 1
				for itl = 1; itl <= 3; itl++ {
					for itr = 1; itr <= 3; itr++ {
						for ib = 1; ib <= 3; ib++ {
							tl.Set(0, 0, val.Get(itl-1))
							tr.Set(0, 0, val.Get(itr-1))
							b.Set(0, 0, val.Get(ib-1))
							(*knt) = (*knt) + 1
							golapack.Dlasy2(ltranl, ltranr, &isgn, &n1, &n2, tl, func() *int { y := 2; return &y }(), tr, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &scale, x, func() *int { y := 2; return &y }(), &xnorm, &info)
							if info != 0 {
								(*ninfo) = (*ninfo) + 1
							}
							res = math.Abs((tl.Get(0, 0)+sgn*tr.Get(0, 0))*x.Get(0, 0) - scale*b.Get(0, 0))
							if info == 0 {
								den = math.Max(eps*((math.Abs(tr.Get(0, 0))+math.Abs(tl.Get(0, 0)))*math.Abs(x.Get(0, 0))), smlnum)
							} else {
								den = smlnum * math.Max(math.Abs(x.Get(0, 0)), one)
							}
							res = res / den
							if scale > one {
								res = res + one/eps
							}
							res = res + math.Abs(xnorm-math.Abs(x.Get(0, 0)))/math.Max(smlnum, xnorm)/eps
							if info != 0 && info != 1 {
								res = res + one/eps
							}
							if res > (*rmax) {
								(*lmax) = (*knt)
								(*rmax) = res
							}
						}
					}
				}

				n1 = 2
				n2 = 1
				for itl = 1; itl <= 8; itl++ {
					for itlscl = 1; itlscl <= 3; itlscl++ {
						for itr = 1; itr <= 3; itr++ {
							for ib1 = 1; ib1 <= 3; ib1++ {
								for ib2 = 1; ib2 <= 3; ib2++ {
									b.Set(0, 0, val.Get(ib1-1))
									b.Set(1, 0, -four*val.Get(ib2-1))
									tl.Set(0, 0, float64(itval[0+(0+(itl-1)*2)*2])*val.Get(itlscl-1))
									tl.Set(1, 0, float64(itval[1+(0+(itl-1)*2)*2])*val.Get(itlscl-1))
									tl.Set(0, 1, float64(itval[0+(1+(itl-1)*2)*2])*val.Get(itlscl-1))
									tl.Set(1, 1, float64(itval[1+(1+(itl-1)*2)*2])*val.Get(itlscl-1))
									tr.Set(0, 0, val.Get(itr-1))
									(*knt) = (*knt) + 1
									golapack.Dlasy2(ltranl, ltranr, &isgn, &n1, &n2, tl, func() *int { y := 2; return &y }(), tr, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &scale, x, func() *int { y := 2; return &y }(), &xnorm, &info)
									if info != 0 {
										(*ninfo) = (*ninfo) + 1
									}
									if ltranl {
										tmp = tl.Get(0, 1)
										tl.Set(0, 1, tl.Get(1, 0))
										tl.Set(1, 0, tmp)
									}
									res = math.Abs((tl.Get(0, 0)+sgn*tr.Get(0, 0))*x.Get(0, 0) + tl.Get(0, 1)*x.Get(1, 0) - scale*b.Get(0, 0))
									res = res + math.Abs((tl.Get(1, 1)+sgn*tr.Get(0, 0))*x.Get(1, 0)+tl.Get(1, 0)*x.Get(0, 0)-scale*b.Get(1, 0))
									tnrm = math.Abs(tr.Get(0, 0)) + math.Abs(tl.Get(0, 0)) + math.Abs(tl.Get(0, 1)) + math.Abs(tl.Get(1, 0)) + math.Abs(tl.Get(1, 1))
									xnrm = math.Max(math.Abs(x.Get(0, 0)), math.Abs(x.Get(1, 0)))
									den = math.Max(smlnum, math.Max(smlnum*xnrm, (tnrm*eps)*xnrm))
									res = res / den
									if scale > one {
										res = res + one/eps
									}
									res = res + math.Abs(xnorm-xnrm)/math.Max(smlnum, xnorm)/eps
									if res > (*rmax) {
										(*lmax) = (*knt)
										(*rmax) = res
									}
								}
							}
						}
					}
				}

				n1 = 1
				n2 = 2
				for itr = 1; itr <= 8; itr++ {
					for itrscl = 1; itrscl <= 3; itrscl++ {
						for itl = 1; itl <= 3; itl++ {
							for ib1 = 1; ib1 <= 3; ib1++ {
								for ib2 = 1; ib2 <= 3; ib2++ {
									b.Set(0, 0, val.Get(ib1-1))
									b.Set(0, 1, -two*val.Get(ib2-1))
									tr.Set(0, 0, float64(itval[0+(0+(itr-1)*2)*2])*val.Get(itrscl-1))
									tr.Set(1, 0, float64(itval[1+(0+(itr-1)*2)*2])*val.Get(itrscl-1))
									tr.Set(0, 1, float64(itval[0+(1+(itr-1)*2)*2])*val.Get(itrscl-1))
									tr.Set(1, 1, float64(itval[1+(1+(itr-1)*2)*2])*val.Get(itrscl-1))
									tl.Set(0, 0, val.Get(itl-1))
									(*knt) = (*knt) + 1
									golapack.Dlasy2(ltranl, ltranr, &isgn, &n1, &n2, tl, func() *int { y := 2; return &y }(), tr, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &scale, x, func() *int { y := 2; return &y }(), &xnorm, &info)
									if info != 0 {
										(*ninfo) = (*ninfo) + 1
									}
									if ltranr {
										tmp = tr.Get(0, 1)
										tr.Set(0, 1, tr.Get(1, 0))
										tr.Set(1, 0, tmp)
									}
									tnrm = math.Abs(tl.Get(0, 0)) + math.Abs(tr.Get(0, 0)) + math.Abs(tr.Get(0, 1)) + math.Abs(tr.Get(1, 1)) + math.Abs(tr.Get(1, 0))
									xnrm = math.Abs(x.Get(0, 0)) + math.Abs(x.Get(0, 1))
									res = math.Abs((tl.Get(0, 0)+sgn*tr.Get(0, 0))*(x.Get(0, 0)) + (sgn*tr.Get(1, 0))*(x.Get(0, 1)) - (scale * b.Get(0, 0)))
									res = res + math.Abs((tl.Get(0, 0)+sgn*tr.Get(1, 1))*(x.Get(0, 1))+(sgn*tr.Get(0, 1))*(x.Get(0, 0))-(scale*b.Get(0, 1)))
									den = math.Max(smlnum, math.Max(smlnum*xnrm, (tnrm*eps)*xnrm))
									res = res / den
									if scale > one {
										res = res + one/eps
									}
									res = res + math.Abs(xnorm-xnrm)/math.Max(smlnum, xnorm)/eps
									if res > (*rmax) {
										(*lmax) = (*knt)
										(*rmax) = res
									}
								}
							}
						}
					}
				}

				n1 = 2
				n2 = 2
				for itr = 1; itr <= 8; itr++ {
					for itrscl = 1; itrscl <= 3; itrscl++ {
						for itl = 1; itl <= 8; itl++ {
							for itlscl = 1; itlscl <= 3; itlscl++ {
								for ib1 = 1; ib1 <= 3; ib1++ {
									for ib2 = 1; ib2 <= 3; ib2++ {
										for ib3 = 1; ib3 <= 3; ib3++ {
											b.Set(0, 0, val.Get(ib1-1))
											b.Set(1, 0, -four*val.Get(ib2-1))
											b.Set(0, 1, -two*val.Get(ib3-1))
											b.Set(1, 1, eight*math.Min(val.Get(ib1-1), math.Min(val.Get(ib2-1), val.Get(ib3-1))))
											tr.Set(0, 0, float64(itval[0+(0+(itr-1)*2)*2])*val.Get(itrscl-1))
											tr.Set(1, 0, float64(itval[1+(0+(itr-1)*2)*2])*val.Get(itrscl-1))
											tr.Set(0, 1, float64(itval[0+(1+(itr-1)*2)*2])*val.Get(itrscl-1))
											tr.Set(1, 1, float64(itval[1+(1+(itr-1)*2)*2])*val.Get(itrscl-1))
											tl.Set(0, 0, float64(itval[0+(0+(itl-1)*2)*2])*val.Get(itlscl-1))
											tl.Set(1, 0, float64(itval[1+(0+(itl-1)*2)*2])*val.Get(itlscl-1))
											tl.Set(0, 1, float64(itval[0+(1+(itl-1)*2)*2])*val.Get(itlscl-1))
											tl.Set(1, 1, float64(itval[1+(1+(itl-1)*2)*2])*val.Get(itlscl-1))
											(*knt) = (*knt) + 1
											golapack.Dlasy2(ltranl, ltranr, &isgn, &n1, &n2, tl, func() *int { y := 2; return &y }(), tr, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &scale, x, func() *int { y := 2; return &y }(), &xnorm, &info)
											if info != 0 {
												(*ninfo) = (*ninfo) + 1
											}
											if ltranr {
												tmp = tr.Get(0, 1)
												tr.Set(0, 1, tr.Get(1, 0))
												tr.Set(1, 0, tmp)
											}
											if ltranl {
												tmp = tl.Get(0, 1)
												tl.Set(0, 1, tl.Get(1, 0))
												tl.Set(1, 0, tmp)
											}
											tnrm = math.Abs(tr.Get(0, 0)) + math.Abs(tr.Get(1, 0)) + math.Abs(tr.Get(0, 1)) + math.Abs(tr.Get(1, 1)) + math.Abs(tl.Get(0, 0)) + math.Abs(tl.Get(1, 0)) + math.Abs(tl.Get(0, 1)) + math.Abs(tl.Get(1, 1))
											xnrm = math.Max(math.Abs(x.Get(0, 0))+math.Abs(x.Get(0, 1)), math.Abs(x.Get(1, 0))+math.Abs(x.Get(1, 1)))
											res = math.Abs((tl.Get(0, 0)+sgn*tr.Get(0, 0))*(x.Get(0, 0)) + (sgn*tr.Get(1, 0))*(x.Get(0, 1)) + (tl.Get(0, 1))*(x.Get(1, 0)) - (scale * b.Get(0, 0)))
											res = res + math.Abs((tl.Get(0, 0))*(x.Get(0, 1))+(sgn*tr.Get(0, 1))*(x.Get(0, 0))+(sgn*tr.Get(1, 1))*(x.Get(0, 1))+(tl.Get(0, 1))*(x.Get(1, 1))-(scale*b.Get(0, 1)))
											res = res + math.Abs((tl.Get(1, 0))*(x.Get(0, 0))+(sgn*tr.Get(0, 0))*(x.Get(1, 0))+(sgn*tr.Get(1, 0))*(x.Get(1, 1))+(tl.Get(1, 1))*(x.Get(1, 0))-(scale*b.Get(1, 0)))
											res = res + math.Abs((tl.Get(1, 1)+sgn*tr.Get(1, 1))*(x.Get(1, 1))+(sgn*tr.Get(0, 1))*(x.Get(1, 0))+(tl.Get(1, 0))*(x.Get(0, 1))-(scale*b.Get(1, 1)))
											den = math.Max(smlnum, math.Max(smlnum*xnrm, (tnrm*eps)*xnrm))
											res = res / den
											if scale > one {
												res = res + one/eps
											}
											res = res + math.Abs(xnorm-xnrm)/math.Max(smlnum, xnorm)/eps
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
}
