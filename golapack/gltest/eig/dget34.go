package eig

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"math"
)

// Dget34 tests DLAEXC, a routine for swapping adjacent blocks (either
// 1 by 1 or 2 by 2) on the diagonal of a matrix in real Schur form.
// Thus, DLAEXC computes an orthogonal matrix Q such that
//
//     Q' * [ A B ] * Q  = [ C1 B1 ]
//          [ 0 C ]        [ 0  A1 ]
//
// where C1 is similar to C and A1 is similar to A.  Both A and C are
// assumed to be in standard form (equal diagonal entries and
// offdiagonal with differing signs) and A1 and C1 are returned with the
// same properties.
//
// The test code verifies these last last assertions, as well as that
// the residual in the above equation is small.
func Dget34(rmax *float64, lmax *int, ninfo *[]int, knt *int) {
	var bignum, eps, half, one, res, smlnum, three, tnrm, two, zero float64
	var i, ia, ia11, ia12, ia21, ia22, iam, ib, ic, ic11, ic12, ic21, ic22, icm, info, j, lwork int

	result := vf(2)
	val := vf(9)
	vm := vf(2)
	work := vf(32)
	q := mf(4, 4, opts)
	t := mf(4, 4, opts)
	t1 := mf(4, 4, opts)

	zero = 0.0
	half = 0.5
	one = 1.0
	two = 2.0
	three = 3.0
	lwork = 32

	//     Get machine parameters
	eps = golapack.Dlamch(Precision)
	smlnum = golapack.Dlamch(SafeMinimum) / eps
	bignum = one / smlnum
	golapack.Dlabad(&smlnum, &bignum)

	//     Set up test case parameters
	val.Set(0, zero)
	val.Set(1, math.Sqrt(smlnum))
	val.Set(2, one)
	val.Set(3, two)
	val.Set(4, math.Sqrt(bignum))
	val.Set(5, -math.Sqrt(smlnum))
	val.Set(6, -one)
	val.Set(7, -two)
	val.Set(8, -math.Sqrt(bignum))
	vm.Set(0, one)
	vm.Set(1, one+two*eps)
	goblas.Dcopy(func() *int { y := 16; return &y }(), val.Off(3), func() *int { y := 0; return &y }(), t.Vector(0, 0), func() *int { y := 1; return &y }())

	(*ninfo)[0] = 0
	(*ninfo)[1] = 0
	(*knt) = 0
	(*lmax) = 0
	(*rmax) = zero

	//     Begin test loop
	for ia = 1; ia <= 9; ia++ {
		for iam = 1; iam <= 2; iam++ {
			for ib = 1; ib <= 9; ib++ {
				for ic = 1; ic <= 9; ic++ {
					t.Set(0, 0, val.Get(ia-1)*vm.Get(iam-1))
					t.Set(1, 1, val.Get(ic-1))
					t.Set(0, 1, val.Get(ib-1))
					t.Set(1, 0, zero)
					tnrm = maxf64(math.Abs(t.Get(0, 0)), math.Abs(t.Get(1, 1)), math.Abs(t.Get(0, 1)))
					goblas.Dcopy(func() *int { y := 16; return &y }(), t.VectorIdx(0), func() *int { y := 1; return &y }(), t1.VectorIdx(0), func() *int { y := 1; return &y }())
					goblas.Dcopy(func() *int { y := 16; return &y }(), val.Off(0), func() *int { y := 0; return &y }(), q.VectorIdx(0), func() *int { y := 1; return &y }())
					goblas.Dcopy(func() *int { y := 4; return &y }(), val.Off(2), func() *int { y := 0; return &y }(), q.VectorIdx(0), func() *int { y := 5; return &y }())
					golapack.Dlaexc(true, func() *int { y := 2; return &y }(), t, func() *int { y := 4; return &y }(), q, func() *int { y := 4; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), work, &info)
					if info != 0 {
						(*ninfo)[info-1] = (*ninfo)[info-1] + 1
					}
					Dhst01(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), t1, func() *int { y := 4; return &y }(), t, func() *int { y := 4; return &y }(), q, func() *int { y := 4; return &y }(), work, &lwork, result)
					res = result.Get(0) + result.Get(1)
					if info != 0 {
						res = res + one/eps
					}
					if t.Get(0, 0) != t1.Get(1, 1) {
						res = res + one/eps
					}
					if t.Get(1, 1) != t1.Get(0, 0) {
						res = res + one/eps
					}
					if t.Get(1, 0) != zero {
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

	for ia = 1; ia <= 5; ia++ {
		for iam = 1; iam <= 2; iam++ {
			for ib = 1; ib <= 5; ib++ {
				for ic11 = 1; ic11 <= 5; ic11++ {
					for ic12 = 2; ic12 <= 5; ic12++ {
						for ic21 = 2; ic21 <= 4; ic21++ {
							for ic22 = -1; ic22 <= 1; ic22 += 2 {
								t.Set(0, 0, val.Get(ia-1)*vm.Get(iam-1))
								t.Set(0, 1, val.Get(ib-1))
								t.Set(0, 2, -two*val.Get(ib-1))
								t.Set(1, 0, zero)
								t.Set(1, 1, val.Get(ic11-1))
								t.Set(1, 2, val.Get(ic12-1))
								t.Set(2, 0, zero)
								t.Set(2, 1, -val.Get(ic21-1))
								t.Set(2, 2, val.Get(ic11-1)*float64(ic22))
								tnrm = maxf64(math.Abs(t.Get(0, 0)), math.Abs(t.Get(0, 1)), math.Abs(t.Get(0, 2)), math.Abs(t.Get(1, 1)), math.Abs(t.Get(1, 2)), math.Abs(t.Get(2, 1)), math.Abs(t.Get(2, 2)))
								goblas.Dcopy(func() *int { y := 16; return &y }(), t.VectorIdx(0), func() *int { y := 1; return &y }(), t1.VectorIdx(0), func() *int { y := 1; return &y }())
								goblas.Dcopy(func() *int { y := 16; return &y }(), val.Off(0), func() *int { y := 0; return &y }(), q.VectorIdx(0), func() *int { y := 1; return &y }())
								goblas.Dcopy(func() *int { y := 4; return &y }(), val.Off(2), func() *int { y := 0; return &y }(), q.VectorIdx(0), func() *int { y := 5; return &y }())
								golapack.Dlaexc(true, func() *int { y := 3; return &y }(), t, func() *int { y := 4; return &y }(), q, func() *int { y := 4; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), work, &info)
								if info != 0 {
									(*ninfo)[info-1] = (*ninfo)[info-1] + 1
								}
								Dhst01(func() *int { y := 3; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 3; return &y }(), t1, func() *int { y := 4; return &y }(), t, func() *int { y := 4; return &y }(), q, func() *int { y := 4; return &y }(), work, &lwork, result)
								res = result.Get(0) + result.Get(1)
								if info == 0 {
									if t1.Get(0, 0) != t.Get(2, 2) {
										res = res + one/eps
									}
									if t.Get(2, 0) != zero {
										res = res + one/eps
									}
									if t.Get(2, 1) != zero {
										res = res + one/eps
									}
									if t.Get(1, 0) != 0 && (t.Get(0, 0) != t.Get(1, 1) || math.Copysign(one, t.Get(0, 1)) == math.Copysign(one, t.Get(1, 0))) {
										res = res + one/eps
									}
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

	for ia11 = 1; ia11 <= 5; ia11++ {
		for ia12 = 2; ia12 <= 5; ia12++ {
			for ia21 = 2; ia21 <= 4; ia21++ {
				for ia22 = -1; ia22 <= 1; ia22 += 2 {
					for icm = 1; icm <= 2; icm++ {
						for ib = 1; ib <= 5; ib++ {
							for ic = 1; ic <= 5; ic++ {
								t.Set(0, 0, val.Get(ia11-1))
								t.Set(0, 1, val.Get(ia12-1))
								t.Set(0, 2, -two*val.Get(ib-1))
								t.Set(1, 0, -val.Get(ia21-1))
								t.Set(1, 1, val.Get(ia11-1)*float64(ia22))
								t.Set(1, 2, val.Get(ib-1))
								t.Set(2, 0, zero)
								t.Set(2, 1, zero)
								t.Set(2, 2, val.Get(ic-1)*vm.Get(icm-1))
								tnrm = maxf64(math.Abs(t.Get(0, 0)), math.Abs(t.Get(0, 1)), math.Abs(t.Get(0, 2)), math.Abs(t.Get(1, 1)), math.Abs(t.Get(1, 2)), math.Abs(t.Get(2, 1)), math.Abs(t.Get(2, 2)))
								goblas.Dcopy(func() *int { y := 16; return &y }(), t.VectorIdx(0), func() *int { y := 1; return &y }(), t1.VectorIdx(0), func() *int { y := 1; return &y }())
								goblas.Dcopy(func() *int { y := 16; return &y }(), val.Off(0), func() *int { y := 0; return &y }(), q.VectorIdx(0), func() *int { y := 1; return &y }())
								goblas.Dcopy(func() *int { y := 4; return &y }(), val.Off(2), func() *int { y := 0; return &y }(), q.VectorIdx(0), func() *int { y := 5; return &y }())
								golapack.Dlaexc(true, func() *int { y := 3; return &y }(), t, func() *int { y := 4; return &y }(), q, func() *int { y := 4; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), work, &info)
								if info != 0 {
									(*ninfo)[info-1] = (*ninfo)[info-1] + 1
								}
								Dhst01(func() *int { y := 3; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 3; return &y }(), t1, func() *int { y := 4; return &y }(), t, func() *int { y := 4; return &y }(), q, func() *int { y := 4; return &y }(), work, &lwork, result)
								res = result.Get(0) + result.Get(1)
								if info == 0 {
									if t1.Get(2, 2) != t.Get(0, 0) {
										res = res + one/eps
									}
									if t.Get(1, 0) != zero {
										res = res + one/eps
									}
									if t.Get(2, 0) != zero {
										res = res + one/eps
									}
									if t.Get(2, 1) != 0 && (t.Get(1, 1) != t.Get(2, 2) || math.Copysign(one, t.Get(1, 2)) == math.Copysign(one, t.Get(2, 1))) {
										res = res + one/eps
									}
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

	for ia11 = 1; ia11 <= 5; ia11++ {
		for ia12 = 2; ia12 <= 5; ia12++ {
			for ia21 = 2; ia21 <= 4; ia21++ {
				for ia22 = -1; ia22 <= 1; ia22 += 2 {
					for ib = 1; ib <= 5; ib++ {
						for ic11 = 3; ic11 <= 4; ic11++ {
							for ic12 = 3; ic12 <= 4; ic12++ {
								for ic21 = 3; ic21 <= 4; ic21++ {
									for ic22 = -1; ic22 <= 1; ic22 += 2 {
										for icm = 5; icm <= 7; icm++ {
											iam = 1
											t.Set(0, 0, val.Get(ia11-1)*vm.Get(iam-1))
											t.Set(0, 1, val.Get(ia12-1)*vm.Get(iam-1))
											t.Set(0, 2, -two*val.Get(ib-1))
											t.Set(0, 3, half*val.Get(ib-1))
											t.Set(1, 0, -t.Get(0, 1)*val.Get(ia21-1))
											t.Set(1, 1, val.Get(ia11-1)*float64(ia22)*vm.Get(iam-1))
											t.Set(1, 2, val.Get(ib-1))
											t.Set(1, 3, three*val.Get(ib-1))
											t.Set(2, 0, zero)
											t.Set(2, 1, zero)
											t.Set(2, 2, val.Get(ic11-1)*math.Abs(val.Get(icm-1)))
											t.Set(2, 3, val.Get(ic12-1)*math.Abs(val.Get(icm-1)))
											t.Set(3, 0, zero)
											t.Set(3, 1, zero)
											t.Set(3, 2, -t.Get(2, 3)*val.Get(ic21-1)*math.Abs(val.Get(icm-1)))
											t.Set(3, 3, val.Get(ic11-1)*float64(ic22)*math.Abs(val.Get(icm-1)))
											tnrm = zero
											for i = 1; i <= 4; i++ {
												for j = 1; j <= 4; j++ {
													tnrm = maxf64(tnrm, math.Abs(t.Get(i-1, j-1)))
												}
											}
											goblas.Dcopy(func() *int { y := 16; return &y }(), t.VectorIdx(0), func() *int { y := 1; return &y }(), t1.VectorIdx(0), func() *int { y := 1; return &y }())
											goblas.Dcopy(func() *int { y := 16; return &y }(), val.Off(0), func() *int { y := 0; return &y }(), q.VectorIdx(0), func() *int { y := 1; return &y }())
											goblas.Dcopy(func() *int { y := 4; return &y }(), val.Off(2), func() *int { y := 0; return &y }(), q.VectorIdx(0), func() *int { y := 5; return &y }())
											golapack.Dlaexc(true, func() *int { y := 4; return &y }(), t, func() *int { y := 4; return &y }(), q, func() *int { y := 4; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), work, &info)
											if info != 0 {
												(*ninfo)[info-1] = (*ninfo)[info-1] + 1
											}
											Dhst01(func() *int { y := 4; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 4; return &y }(), t1, func() *int { y := 4; return &y }(), t, func() *int { y := 4; return &y }(), q, func() *int { y := 4; return &y }(), work, &lwork, result)
											res = result.Get(0) + result.Get(1)
											if info == 0 {
												if t.Get(2, 0) != zero {
													res = res + one/eps
												}
												if t.Get(3, 0) != zero {
													res = res + one/eps
												}
												if t.Get(2, 1) != zero {
													res = res + one/eps
												}
												if t.Get(3, 1) != zero {
													res = res + one/eps
												}
												if t.Get(1, 0) != 0 && (t.Get(0, 0) != t.Get(1, 1) || math.Copysign(one, t.Get(0, 1)) == math.Copysign(one, t.Get(1, 0))) {
													res = res + one/eps
												}
												if t.Get(3, 2) != 0 && (t.Get(2, 2) != t.Get(3, 3) || math.Copysign(one, t.Get(2, 3)) == math.Copysign(one, t.Get(3, 2))) {
													res = res + one/eps
												}
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
}