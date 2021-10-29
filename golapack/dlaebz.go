package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlaebz contains the iteration loops which compute and use the
// function N(w), which is the count of eigenvalues of a symmetric
// tridiagonal matrix T less than or equal to its argument  w.  It
// performs a choice of two types of loops:
//
// IJOB=1, followed by
// IJOB=2: It takes as input a list of intervals and returns a list of
//         sufficiently small intervals whose union contains the same
//         eigenvalues as the union of the original intervals.
//         The input intervals are (AB(j,1),AB(j,2)], j=1,...,MINP.
//         The output interval (AB(j,1),AB(j,2)] will contain
//         eigenvalues NAB(j,1)+1,...,NAB(j,2), where 1 <= j <= MOUT.
//
// IJOB=3: It performs a binary search in each input interval
//         (AB(j,1),AB(j,2)] for a point  w(j)  such that
//         N(w(j))=NVAL(j), and uses  C(j)  as the starting point of
//         the search.  If such a w(j) is found, then on output
//         AB(j,1)=AB(j,2)=w.  If no such w(j) is found, then on output
//         (AB(j,1),AB(j,2)] will be a small interval containing the
//         point where N(w) jumps through NVAL(j), unless that point
//         lies outside the initial interval.
//
// Note that the intervals are in all cases half-open intervals,
// i.e., of the form  (a,b] , which includes  b  but not  a .
//
// To avoid underflow, the matrix should be scaled so that its largest
// element is no greater than  overflow**(1/2) * underflow**(1/4)
// in absolute value.  To assure the most accurate computation
// of small eigenvalues, the matrix should be scaled to be
// not much smaller than that, either.
//
// See W. Kahan "Accurate Eigenvalues of a Symmetric Tridiagonal
// Matrix", Report CS41, Computer Science Dept., Stanford
// University, July 21, 1966
//
// Note: the arguments are, in general, *not* checked for unreasonable
// values.
func Dlaebz(ijob, nitmax, n, mmax, minp, nbmin int, abstol, reltol, pivmin float64, d, e, e2 *mat.Vector, nval *[]int, ab *mat.Matrix, c *mat.Vector, nab *[]int, work *mat.Vector, iwork *[]int) (mout, info int, err error) {
	var half, tmp1, tmp2, two, zero float64
	var itmp1, itmp2, j, ji, jit, jp, kf, kfnew, kl, klnew int

	zero = 0.0
	two = 2.0
	half = 1.0 / two

	//     Check for Errors
	if ijob < 1 || ijob > 3 {
		err = fmt.Errorf("ijob < 1 || ijob > 3: ijob=%v", ijob)
		return
	}

	//     Initialize NAB
	if ijob == 1 {
		//        Compute the number of eigenvalues in the initial intervals.
		mout = 0
		for ji = 1; ji <= minp; ji++ {
			for jp = 1; jp <= 2; jp++ {
				tmp1 = d.Get(0) - ab.Get(ji-1, jp-1)
				if math.Abs(tmp1) < pivmin {
					tmp1 = -pivmin
				}
				(*nab)[ji-1+(jp-1)*mmax] = 0
				if tmp1 <= zero {
					(*nab)[ji-1+(jp-1)*mmax] = 1
				}

				for j = 2; j <= n; j++ {
					tmp1 = d.Get(j-1) - e2.Get(j-1-1)/tmp1 - ab.Get(ji-1, jp-1)
					if math.Abs(tmp1) < pivmin {
						tmp1 = -pivmin
					}
					if tmp1 <= zero {
						(*nab)[ji-1+(jp-1)*mmax] = (*nab)[ji-1+(jp-1)*mmax] + 1
					}
				}
			}
			mout = mout + (*nab)[ji-1+(1)*mmax] - (*nab)[ji-1+(0)*mmax]
		}
		return
	}

	//     Initialize for loop
	//
	//     KF and KL have the following meaning:
	//        Intervals 1,...,KF-1 have converged.
	//        Intervals KF,...,KL  still need to be refined.
	kf = 1
	kl = minp

	//     If IJOB=2, initialize C.
	//     If IJOB=3, use the user-supplied starting point.
	if ijob == 2 {
		for ji = 1; ji <= minp; ji++ {
			c.Set(ji-1, half*(ab.Get(ji-1, 0)+ab.Get(ji-1, 1)))
		}
	}

	//     Iteration loop
	for jit = 1; jit <= nitmax; jit++ {
		//        Loop over intervals
		if kl-kf+1 >= nbmin && nbmin > 0 {
			//           Begin of Parallel Version of the loop
			for ji = kf; ji <= kl; ji++ {
				//              Compute N(c), the number of eigenvalues less than c
				work.Set(ji-1, d.Get(0)-c.Get(ji-1))
				(*iwork)[ji-1] = 0
				if work.Get(ji-1) <= pivmin {
					(*iwork)[ji-1] = 1
					work.Set(ji-1, math.Min(work.Get(ji-1), -pivmin))
				}

				for j = 2; j <= n; j++ {
					work.Set(ji-1, d.Get(j-1)-e2.Get(j-1-1)/work.Get(ji-1)-c.Get(ji-1))
					if work.Get(ji-1) <= pivmin {
						(*iwork)[ji-1] = (*iwork)[ji-1] + 1
						work.Set(ji-1, math.Min(work.Get(ji-1), -pivmin))
					}
				}
			}

			if ijob <= 2 {
				//              IJOB=2: Choose all intervals containing eigenvalues.
				klnew = kl
				for ji = kf; ji <= kl; ji++ {
					//                 Insure that N(w) is monotone
					(*iwork)[ji-1] = min((*nab)[ji-1+(1)*mmax], max((*nab)[ji-1+(0)*mmax], (*iwork)[ji-1]))

					//                 Update the Queue -- add intervals if both halves
					//                 contain eigenvalues.
					if (*iwork)[ji-1] == (*nab)[ji-1+(1)*mmax] {
						//                    No eigenvalue in the upper interval:
						//                    just use the lower interval.
						ab.Set(ji-1, 1, c.Get(ji-1))

					} else if (*iwork)[ji-1] == (*nab)[ji-1+(0)*mmax] {
						//                    No eigenvalue in the lower interval:
						//                    just use the upper interval.
						ab.Set(ji-1, 0, c.Get(ji-1))
					} else {
						klnew = klnew + 1
						if klnew <= mmax {
							//                       Eigenvalue in both intervals -- add upper to
							//                       queue.
							ab.Set(klnew-1, 1, ab.Get(ji-1, 1))
							(*nab)[klnew-1+(1)*mmax] = (*nab)[ji-1+(1)*mmax]
							ab.Set(klnew-1, 0, c.Get(ji-1))
							(*nab)[klnew-1+(0)*mmax] = (*iwork)[ji-1]
							ab.Set(ji-1, 1, c.Get(ji-1))
							(*nab)[ji-1+(1)*mmax] = (*iwork)[ji-1]
						} else {
							info = mmax + 1
						}
					}
				}
				if info != 0 {
					return
				}
				kl = klnew
			} else {
				//              IJOB=3: Binary search.  Keep only the interval containing
				//                      w   s.t. N(w) = NVAL
				for ji = kf; ji <= kl; ji++ {
					if (*iwork)[ji-1] <= (*nval)[ji-1] {
						ab.Set(ji-1, 0, c.Get(ji-1))
						(*nab)[ji-1+(0)*mmax] = (*iwork)[ji-1]
					}
					if (*iwork)[ji-1] >= (*nval)[ji-1] {
						ab.Set(ji-1, 1, c.Get(ji-1))
						(*nab)[ji-1+(1)*mmax] = (*iwork)[ji-1]
					}
				}
			}

		} else {
			//           End of Parallel Version of the loop
			//
			//           Begin of Serial Version of the loop
			klnew = kl
			for ji = kf; ji <= kl; ji++ {
				//              Compute N(w), the number of eigenvalues less than w
				tmp1 = c.Get(ji - 1)
				tmp2 = d.Get(0) - tmp1
				itmp1 = 0
				if tmp2 <= pivmin {
					itmp1 = 1
					tmp2 = math.Min(tmp2, -pivmin)
				}

				for j = 2; j <= n; j++ {
					tmp2 = d.Get(j-1) - e2.Get(j-1-1)/tmp2 - tmp1
					if tmp2 <= pivmin {
						itmp1 = itmp1 + 1
						tmp2 = math.Min(tmp2, -pivmin)
					}
				}

				if ijob <= 2 {
					//                 IJOB=2: Choose all intervals containing eigenvalues.
					//
					//                 Insure that N(w) is monotone
					itmp1 = min((*nab)[ji-1+(1)*mmax], max((*nab)[ji-1+(0)*mmax], itmp1))

					//                 Update the Queue -- add intervals if both halves
					//                 contain eigenvalues.
					if itmp1 == (*nab)[ji-1+(1)*mmax] {
						//                    No eigenvalue in the upper interval:
						//                    just use the lower interval.
						ab.Set(ji-1, 1, tmp1)

					} else if itmp1 == (*nab)[ji-1+(0)*mmax] {
						//                    No eigenvalue in the lower interval:
						//                    just use the upper interval.
						ab.Set(ji-1, 0, tmp1)
					} else if klnew < mmax {
						//                    Eigenvalue in both intervals -- add upper to queue.
						klnew = klnew + 1
						ab.Set(klnew-1, 1, ab.Get(ji-1, 1))
						(*nab)[klnew-1+(1)*mmax] = (*nab)[ji-1+(1)*mmax]
						ab.Set(klnew-1, 0, tmp1)
						(*nab)[klnew-1+(0)*mmax] = itmp1
						ab.Set(ji-1, 1, tmp1)
						(*nab)[ji-1+(1)*mmax] = itmp1
					} else {
						info = mmax + 1
						return
					}
				} else {
					//                 IJOB=3: Binary search.  Keep only the interval
					//                         containing  w  s.t. N(w) = NVAL
					if itmp1 <= (*nval)[ji-1] {
						ab.Set(ji-1, 0, tmp1)
						(*nab)[ji-1+(0)*mmax] = itmp1
					}
					if itmp1 >= (*nval)[ji-1] {
						ab.Set(ji-1, 1, tmp1)
						(*nab)[ji-1+(1)*mmax] = itmp1
					}
				}
			}
			kl = klnew

		}

		//        Check for convergence
		kfnew = kf
		for ji = kf; ji <= kl; ji++ {
			tmp1 = math.Abs(ab.Get(ji-1, 1) - ab.Get(ji-1, 0))
			tmp2 = math.Max(math.Abs(ab.Get(ji-1, 1)), math.Abs(ab.Get(ji-1, 0)))
			if tmp1 < math.Max(abstol, math.Max(pivmin, reltol*tmp2)) || (*nab)[ji-1+(0)*mmax] >= (*nab)[ji-1+(1)*mmax] {
				//              Converged -- Swap with position KFNEW,
				//                           then increment KFNEW
				if ji > kfnew {
					tmp1 = ab.Get(ji-1, 0)
					tmp2 = ab.Get(ji-1, 1)
					itmp1 = (*nab)[ji-1+(0)*mmax]
					itmp2 = (*nab)[ji-1+(1)*mmax]
					ab.Set(ji-1, 0, ab.Get(kfnew-1, 0))
					ab.Set(ji-1, 1, ab.Get(kfnew-1, 1))
					(*nab)[ji-1+(0)*mmax] = (*nab)[kfnew-1+(0)*mmax]
					(*nab)[ji-1+(1)*mmax] = (*nab)[kfnew-1+(1)*mmax]
					ab.Set(kfnew-1, 0, tmp1)
					ab.Set(kfnew-1, 1, tmp2)
					(*nab)[kfnew-1+(0)*mmax] = itmp1
					(*nab)[kfnew-1+(1)*mmax] = itmp2
					if ijob == 3 {
						itmp1 = (*nval)[ji-1]
						(*nval)[ji-1] = (*nval)[kfnew-1]
						(*nval)[kfnew-1] = itmp1
					}
				}
				kfnew = kfnew + 1
			}
		}
		kf = kfnew

		//        Choose Midpoints
		for ji = kf; ji <= kl; ji++ {
			c.Set(ji-1, half*(ab.Get(ji-1, 0)+ab.Get(ji-1, 1)))
		}

		//        If no more intervals to refine, quit.
		if kf > kl {
			goto label140
		}
	}

	//     Converged
label140:
	;
	info = max(kl+1-kf, 0)
	mout = kl

	return
}
