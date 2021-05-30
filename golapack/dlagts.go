package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dlagts may be used to solve one of the systems of equations
//
//    (T - lambda*I)*x = y   or   (T - lambda*I)**T*x = y,
//
// where T is an n by n tridiagonal matrix, for x, following the
// factorization of (T - lambda*I) as
//
//    (T - lambda*I) = P*L*U ,
//
// by routine DLAGTF. The choice of equation to be solved is
// controlled by the argument JOB, and in each case there is an option
// to perturb zero or very small diagonal elements of U, this option
// being intended for use in applications such as inverse iteration.
func Dlagts(job, n *int, a, b, c, d *mat.Vector, in *[]int, y *mat.Vector, tol *float64, info *int) {
	var absak, ak, bignum, eps, one, pert, sfmin, temp, zero float64
	var k int

	one = 1.0
	zero = 0.0

	(*info) = 0
	if (absint(*job) > 2) || ((*job) == 0) {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLAGTS"), -(*info))
		return
	}

	if (*n) == 0 {
		return
	}

	eps = Dlamch(Epsilon)
	sfmin = Dlamch(SafeMinimum)
	bignum = one / sfmin

	if (*job) < 0 {
		if (*tol) <= zero {
			(*tol) = math.Abs(a.Get(0))
			if (*n) > 1 {
				(*tol) = maxf64(*tol, math.Abs(a.Get(1)), math.Abs(b.Get(0)))
			}
			for k = 3; k <= (*n); k++ {
				(*tol) = maxf64(*tol, math.Abs(a.Get(k-1)), math.Abs(b.Get(k-1-1)), math.Abs(d.Get(k-2-1)))
			}
			(*tol) = (*tol) * eps
			if (*tol) == zero {
				(*tol) = eps
			}
		}
	}

	if absint(*job) == 1 {
		for k = 2; k <= (*n); k++ {
			if (*in)[k-1-1] == 0 {
				y.Set(k-1, y.Get(k-1)-c.Get(k-1-1)*y.Get(k-1-1))
			} else {
				temp = y.Get(k - 1 - 1)
				y.Set(k-1-1, y.Get(k-1))
				y.Set(k-1, temp-c.Get(k-1-1)*y.Get(k-1))
			}
		}
		if (*job) == 1 {
			for k = (*n); k >= 1; k-- {
				if k <= (*n)-2 {
					temp = y.Get(k-1) - b.Get(k-1)*y.Get(k+1-1) - d.Get(k-1)*y.Get(k+2-1)
				} else if k == (*n)-1 {
					temp = y.Get(k-1) - b.Get(k-1)*y.Get(k+1-1)
				} else {
					temp = y.Get(k - 1)
				}
				ak = a.Get(k - 1)
				absak = math.Abs(ak)
				if absak < one {
					if absak < sfmin {
						if absak == zero || math.Abs(temp)*sfmin > absak {
							(*info) = k
							return
						} else {
							temp = temp * bignum
							ak = ak * bignum
						}
					} else if math.Abs(temp) > absak*bignum {
						(*info) = k
						return
					}
				}
				y.Set(k-1, temp/ak)
			}
		} else {
			for k = (*n); k >= 1; k-- {
				if k <= (*n)-2 {
					temp = y.Get(k-1) - b.Get(k-1)*y.Get(k+1-1) - d.Get(k-1)*y.Get(k+2-1)
				} else if k == (*n)-1 {
					temp = y.Get(k-1) - b.Get(k-1)*y.Get(k+1-1)
				} else {
					temp = y.Get(k - 1)
				}
				ak = a.Get(k - 1)
				pert = signf64(*tol, ak)
			label40:
				;
				absak = math.Abs(ak)
				if absak < one {
					if absak < sfmin {
						if absak == zero || math.Abs(temp)*sfmin > absak {
							ak = ak + pert
							pert = 2 * pert
							goto label40
						} else {
							temp = temp * bignum
							ak = ak * bignum
						}
					} else if math.Abs(temp) > absak*bignum {
						ak = ak + pert
						pert = 2 * pert
						goto label40
					}
				}
				y.Set(k-1, temp/ak)
			}
		}
	} else {
		//        Come to here if  JOB = 2 or -2
		if (*job) == 2 {
			for k = 1; k <= (*n); k++ {
				if k >= 3 {
					temp = y.Get(k-1) - b.Get(k-1-1)*y.Get(k-1-1) - d.Get(k-2-1)*y.Get(k-2-1)
				} else if k == 2 {
					temp = y.Get(k-1) - b.Get(k-1-1)*y.Get(k-1-1)
				} else {
					temp = y.Get(k - 1)
				}
				ak = a.Get(k - 1)
				absak = math.Abs(ak)
				if absak < one {
					if absak < sfmin {
						if absak == zero || math.Abs(temp)*sfmin > absak {
							(*info) = k
							return
						} else {
							temp = temp * bignum
							ak = ak * bignum
						}
					} else if math.Abs(temp) > absak*bignum {
						(*info) = k
						return
					}
				}
				y.Set(k-1, temp/ak)
			}
		} else {
			for k = 1; k <= (*n); k++ {
				if k >= 3 {
					temp = y.Get(k-1) - b.Get(k-1-1)*y.Get(k-1-1) - d.Get(k-2-1)*y.Get(k-2-1)
				} else if k == 2 {
					temp = y.Get(k-1) - b.Get(k-1-1)*y.Get(k-1-1)
				} else {
					temp = y.Get(k - 1)
				}
				ak = a.Get(k - 1)
				pert = signf64(*tol, ak)
			label70:
				;
				absak = math.Abs(ak)
				if absak < one {
					if absak < sfmin {
						if absak == zero || math.Abs(temp)*sfmin > absak {
							ak = ak + pert
							pert = 2 * pert
							goto label70
						} else {
							temp = temp * bignum
							ak = ak * bignum
						}
					} else if math.Abs(temp) > absak*bignum {
						ak = ak + pert
						pert = 2 * pert
						goto label70
					}
				}
				y.Set(k-1, temp/ak)
			}
		}

		for k = (*n); k >= 2; k-- {
			if (*in)[k-1-1] == 0 {
				y.Set(k-1-1, y.Get(k-1-1)-c.Get(k-1-1)*y.Get(k-1))
			} else {
				temp = y.Get(k - 1 - 1)
				y.Set(k-1-1, y.Get(k-1))
				y.Set(k-1, temp-c.Get(k-1-1)*y.Get(k-1))
			}
		}
	}
}
