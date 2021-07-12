package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dlarrf Given the initial representation L D L^T and its cluster of close
// eigenvalues (in a relative measure), W( CLSTRT ), W( CLSTRT+1 ), ...
// W( CLEND ), DLARRF finds a new relatively robust representation
// L D L^T - SIGMA I = L(+) D(+) L(+)^T such that at least one of the
// eigenvalues of L(+) D(+) L(+)^T is relatively isolated.
func Dlarrf(n *int, d, l, ld *mat.Vector, clstrt, clend *int, w, wgap, werr *mat.Vector, spdiam, clgapl, clgapr, pivmin, sigma *float64, dplus, lplus, work *mat.Vector, info *int) {
	var dorrr1, forcer, nofail, sawnan1, sawnan2, tryrrr1 bool
	var avgap, bestshift, clwdth, eps, fact, fail, fail2, four, growthbound, ldelta, ldmax, lsigma, max1, max2, maxgrowth1, maxgrowth2, mingap, oldp, one, prod, quart, rdelta, rdmax, rrr1, rrr2, rsigma, s, smlgrowth, tmp, two, znm2 float64
	var i, indx, ktry, ktrymax, shift, sleft, sright int

	one = 1.0
	two = 2.0
	four = 4.0
	quart = 0.25
	maxgrowth1 = 8.
	maxgrowth2 = 8.

	ktrymax = 1
	sleft = 1
	sright = 2

	(*info) = 0

	//     Quick return if possible
	if (*n) <= 0 {
		return
	}

	fact = math.Pow(2, float64(ktrymax))
	eps = Dlamch(Precision)
	shift = 0
	forcer = false
	//     Note that we cannot guarantee that for any of the shifts tried,
	//     the factorization has a small or even moderate element growth.
	//     There could be Ritz values at both ends of the cluster and despite
	//     backing off, there are examples where all factorizations tried
	//     (in IEEE mode, allowing zero pivots & infinities) have INFINITE
	//     element growth.
	//     For this reason, we should use PIVMIN in this subroutine so that at
	//     least the L D L^T factorization exists. It can be checked afterwards
	//     whether the element growth caused bad residuals/orthogonality.
	//     Decide whether the code should accept the best among all
	//     representations despite large element growth or signal INFO=1
	//     Setting NOFAIL to .FALSE. for quick fix for bug 113
	nofail = false

	//     Compute the average gap length of the cluster
	clwdth = math.Abs(w.Get((*clend)-1)-w.Get((*clstrt)-1)) + werr.Get((*clend)-1) + werr.Get((*clstrt)-1)
	avgap = clwdth / float64((*clend)-(*clstrt))
	mingap = math.Min(*clgapl, *clgapr)
	//     Initial values for shifts to both ends of cluster
	lsigma = math.Min(w.Get((*clstrt)-1), w.Get((*clend)-1)) - werr.Get((*clstrt)-1)
	rsigma = math.Max(w.Get((*clstrt)-1), w.Get((*clend)-1)) + werr.Get((*clend)-1)
	//     Use a small fudge to make sure that we really shift to the outside
	lsigma = lsigma - math.Abs(lsigma)*four*eps
	rsigma = rsigma + math.Abs(rsigma)*four*eps
	//     Compute upper bounds for how much to back off the initial shifts
	ldmax = quart*mingap + two*(*pivmin)
	rdmax = quart*mingap + two*(*pivmin)
	ldelta = math.Max(avgap, wgap.Get((*clstrt)-1)) / fact
	rdelta = math.Max(avgap, wgap.Get((*clend)-1-1)) / fact

	//     Initialize the record of the best representation found
	s = Dlamch(SafeMinimum)
	smlgrowth = one / s
	fail = float64((*n)-1) * mingap / ((*spdiam) * eps)
	fail2 = float64((*n)-1) * mingap / ((*spdiam) * math.Sqrt(eps))
	bestshift = lsigma

	//     while (KTRY <= KTRYMAX)
	ktry = 0
	growthbound = maxgrowth1 * (*spdiam)
label5:
	;
	sawnan1 = false
	sawnan2 = false
	//     Ensure that we do not back off too much of the initial shifts
	ldelta = math.Min(ldmax, ldelta)
	rdelta = math.Min(rdmax, rdelta)
	//     Compute the element growth when shifting to both ends of the cluster
	//     accept the shift if there is no element growth at one of the two ends
	//     Left end
	s = -lsigma
	dplus.Set(0, d.Get(0)+s)
	if math.Abs(dplus.Get(0)) < (*pivmin) {
		dplus.Set(0, -(*pivmin))
		//        Need to set SAWNAN1 because refined RRR test should not be used
		//        in this case
		sawnan1 = true
	}
	max1 = math.Abs(dplus.Get(0))
	for i = 1; i <= (*n)-1; i++ {
		lplus.Set(i-1, ld.Get(i-1)/dplus.Get(i-1))
		s = s*lplus.Get(i-1)*l.Get(i-1) - lsigma
		dplus.Set(i, d.Get(i)+s)
		if math.Abs(dplus.Get(i)) < (*pivmin) {
			dplus.Set(i, -(*pivmin))
			//           Need to set SAWNAN1 because refined RRR test should not be used
			//           in this case
			sawnan1 = true
		}
		max1 = math.Max(max1, math.Abs(dplus.Get(i)))
	}
	sawnan1 = sawnan1 || Disnan(int(max1))
	if forcer || (max1 <= growthbound && !sawnan1) {
		(*sigma) = lsigma
		shift = sleft
		goto label100
	}
	//     Right end
	s = -rsigma
	work.Set(0, d.Get(0)+s)
	if math.Abs(work.Get(0)) < (*pivmin) {
		work.Set(0, -(*pivmin))
		//        Need to set SAWNAN2 because refined RRR test should not be used
		//        in this case
		sawnan2 = true
	}
	max2 = math.Abs(work.Get(0))
	for i = 1; i <= (*n)-1; i++ {
		work.Set((*n)+i-1, ld.Get(i-1)/work.Get(i-1))
		s = s*work.Get((*n)+i-1)*l.Get(i-1) - rsigma
		work.Set(i, d.Get(i)+s)
		if math.Abs(work.Get(i)) < (*pivmin) {
			work.Set(i, -(*pivmin))
			//           Need to set SAWNAN2 because refined RRR test should not be used
			//           in this case
			sawnan2 = true
		}
		max2 = math.Max(max2, math.Abs(work.Get(i)))
	}
	sawnan2 = sawnan2 || Disnan(int(max2))
	if forcer || (max2 <= growthbound && !sawnan2) {
		(*sigma) = rsigma
		shift = sright
		goto label100
	}
	//     If we are at this point, both shifts led to too much element growth
	//     Record the better of the two shifts (provided it didn't lead to NaN)
	if sawnan1 && sawnan2 {
		//        both MAX1 and MAX2 are NaN
		goto label50
	} else {
		if !sawnan1 {
			indx = 1
			if max1 <= smlgrowth {
				smlgrowth = max1
				bestshift = lsigma
			}
		}
		if !sawnan2 {
			if sawnan1 || max2 <= max1 {
				indx = 2
			}
			if max2 <= smlgrowth {
				smlgrowth = max2
				bestshift = rsigma
			}
		}
	}
	//     If we are here, both the left and the right shift led to
	//     element growth. If the element growth is moderate, then
	//     we may still accept the representation, if it passes a
	//     refined test for RRR. This test supposes that no NaN occurred.
	//     Moreover, we use the refined RRR test only for isolated clusters.
	if (clwdth < mingap/float64(128)) && (math.Min(max1, max2) < fail2) && (!sawnan1) && (!sawnan2) {
		dorrr1 = true
	} else {
		dorrr1 = false
	}
	tryrrr1 = true
	if tryrrr1 && dorrr1 {
		if indx == 1 {
			tmp = math.Abs(dplus.Get((*n) - 1))
			znm2 = one
			prod = one
			oldp = one
			for i = (*n) - 1; i >= 1; i-- {
				if prod <= eps {
					prod = ((dplus.Get(i) * work.Get((*n)+i)) / (dplus.Get(i-1) * work.Get((*n)+i-1))) * oldp
				} else {
					prod = prod * math.Abs(work.Get((*n)+i-1))
				}
				oldp = prod
				znm2 = znm2 + math.Pow(prod, 2)
				tmp = math.Max(tmp, math.Abs(dplus.Get(i-1)*prod))
			}
			rrr1 = tmp / ((*spdiam) * math.Sqrt(znm2))
			if rrr1 <= maxgrowth2 {
				(*sigma) = lsigma
				shift = sleft
				goto label100
			}
		} else if indx == 2 {
			tmp = math.Abs(work.Get((*n) - 1))
			znm2 = one
			prod = one
			oldp = one
			for i = (*n) - 1; i >= 1; i-- {
				if prod <= eps {
					prod = ((work.Get(i) * lplus.Get(i)) / (work.Get(i-1) * lplus.Get(i-1))) * oldp
				} else {
					prod = prod * math.Abs(lplus.Get(i-1))
				}
				oldp = prod
				znm2 = znm2 + math.Pow(prod, 2)
				tmp = math.Max(tmp, math.Abs(work.Get(i-1)*prod))
			}
			rrr2 = tmp / ((*spdiam) * math.Sqrt(znm2))
			if rrr2 <= maxgrowth2 {
				(*sigma) = rsigma
				shift = sright
				goto label100
			}
		}
	}
label50:
	;
	if ktry < ktrymax {
		//        If we are here, both shifts failed also the RRR test.
		//        Back off to the outside
		lsigma = math.Max(lsigma-ldelta, lsigma-ldmax)
		rsigma = math.Min(rsigma+rdelta, rsigma+rdmax)
		ldelta = two * ldelta
		rdelta = two * rdelta
		ktry = ktry + 1
		goto label5
	} else {
		//        None of the representations investigated satisfied our
		//        criteria. Take the best one we found.
		if (smlgrowth < fail) || nofail {
			lsigma = bestshift
			rsigma = bestshift
			forcer = true
			goto label5
		} else {
			(*info) = 1
			return
		}
	}
label100:
	;
	if shift == sleft {
	} else if shift == sright {
		//        store new L and D back into DPLUS, LPLUS
		goblas.Dcopy(*n, work.Off(0, 1), dplus.Off(0, 1))
		goblas.Dcopy((*n)-1, work.Off((*n), 1), lplus.Off(0, 1))
	}
}
