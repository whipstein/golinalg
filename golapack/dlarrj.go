package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlarrj Given the initial eigenvalue approximations of T, DLARRJ
// does  bisection to refine the eigenvalues of T,
// W( IFIRST-OFFSET ) through W( ILAST-OFFSET ), to more accuracy. Initial
// guesses for these eigenvalues are input in W, the corresponding estimate
// of the error in these guesses in WERR. During bisection, intervals
// [left, right] are maintained by storing their mid-points and
// semi-widths in the arrays W and WERR respectively.
func Dlarrj(n *int, d, e2 *mat.Vector, ifirst, ilast *int, rtol *float64, offset *int, w, werr, work *mat.Vector, iwork *[]int, pivmin, spdiam *float64, info *int) {
	var dplus, fac, half, left, mid, one, right, s, tmp, two, width, zero float64
	var cnt, i, i1, i2, ii, iter, j, k, maxitr, next, nint, olnint, p, prev, savi1 int

	zero = 0.0
	one = 1.0
	two = 2.0
	half = 0.5

	(*info) = 0

	//     Quick return if possible
	if (*n) <= 0 {
		return
	}

	maxitr = int((math.Log((*spdiam)+(*pivmin))-math.Log(*pivmin))/math.Log(two)) + 2

	//     Initialize unconverged intervals in [ WORK(2*I-1), WORK(2*I) ].
	//     The Sturm Count, Count( WORK(2*I-1) ) is arranged to be I-1, while
	//     Count( WORK(2*I) ) is stored in IWORK( 2*I ). The integer IWORK( 2*I-1 )
	//     for an unconverged interval is set to the index of the next unconverged
	//     interval, and is -1 or 0 for a converged interval. Thus a linked
	//     list of unconverged intervals is set up.
	i1 = (*ifirst)
	i2 = (*ilast)
	//     The number of unconverged intervals
	nint = 0
	//     The last unconverged interval found
	prev = 0
	for i = i1; i <= i2; i++ {
		k = 2 * i
		ii = i - (*offset)
		left = w.Get(ii-1) - werr.Get(ii-1)
		mid = w.Get(ii - 1)
		right = w.Get(ii-1) + werr.Get(ii-1)
		width = right - mid
		tmp = math.Max(math.Abs(left), math.Abs(right))
		//        The following test prevents the test of converged intervals
		if width < (*rtol)*tmp {
			//           This interval has already converged and does not need refinement.
			//           (Note that the gaps might change through refining the
			//            eigenvalues, however, they can only get bigger.)
			//           Remove it from the list.
			(*iwork)[k-1-1] = -1
			//           Make sure that I1 always points to the first unconverged interval
			if (i == i1) && (i < i2) {
				i1 = i + 1
			}
			if (prev >= i1) && (i <= i2) {
				(*iwork)[2*prev-1-1] = i + 1
			}
		} else {
			//           unconverged interval found
			prev = i
			//           Make sure that [LEFT,RIGHT] contains the desired eigenvalue
			//
			//           Do while( CNT(LEFT).GT.I-1 )

			fac = one
		label20:
			;
			cnt = 0
			s = left
			dplus = d.Get(0) - s
			if dplus < zero {
				cnt = cnt + 1
			}
			for j = 2; j <= (*n); j++ {
				dplus = d.Get(j-1) - s - e2.Get(j-1-1)/dplus
				if dplus < zero {
					cnt = cnt + 1
				}
			}
			if cnt > i-1 {
				left = left - werr.Get(ii-1)*fac
				fac = two * fac
				goto label20
			}

			//           Do while( CNT(RIGHT).LT.I )
			fac = one
		label50:
			;
			cnt = 0
			s = right
			dplus = d.Get(0) - s
			if dplus < zero {
				cnt = cnt + 1
			}
			for j = 2; j <= (*n); j++ {
				dplus = d.Get(j-1) - s - e2.Get(j-1-1)/dplus
				if dplus < zero {
					cnt = cnt + 1
				}
			}
			if cnt < i {
				right = right + werr.Get(ii-1)*fac
				fac = two * fac
				goto label50
			}
			nint = nint + 1
			(*iwork)[k-1-1] = i + 1
			(*iwork)[k-1] = cnt
		}
		work.Set(k-1-1, left)
		work.Set(k-1, right)
	}
	savi1 = i1

	//     Do while( NINT.GT.0 ), i.e. there are still unconverged intervals
	//     and while (ITER.LT.MAXITR)
	iter = 0
label80:
	;
	prev = i1 - 1
	i = i1
	olnint = nint
	for p = 1; p <= olnint; p++ {
		k = 2 * i
		ii = i - (*offset)
		next = (*iwork)[k-1-1]
		left = work.Get(k - 1 - 1)
		right = work.Get(k - 1)
		mid = half * (left + right)
		//        semiwidth of interval
		width = right - mid
		tmp = math.Max(math.Abs(left), math.Abs(right))
		if (width < (*rtol)*tmp) || (iter == maxitr) {
			//           reduce number of unconverged intervals
			nint = nint - 1
			//           Mark interval as converged.
			(*iwork)[k-1-1] = 0
			if i1 == i {
				i1 = next
			} else {
				//              Prev holds the last unconverged interval previously examined
				if prev >= i1 {
					(*iwork)[2*prev-1-1] = next
				}
			}
			i = next
			goto label100
		}
		prev = i

		//        Perform one bisection step
		cnt = 0
		s = mid
		dplus = d.Get(0) - s
		if dplus < zero {
			cnt = cnt + 1
		}
		for j = 2; j <= (*n); j++ {
			dplus = d.Get(j-1) - s - e2.Get(j-1-1)/dplus
			if dplus < zero {
				cnt = cnt + 1
			}
		}
		if cnt <= i-1 {
			work.Set(k-1-1, mid)
		} else {
			work.Set(k-1, mid)
		}
		i = next
	label100:
	}
	iter = iter + 1
	//     do another loop if there are still unconverged intervals
	//     However, in the last iteration, all intervals are accepted
	//     since this is the best we can do.
	if (nint > 0) && (iter <= maxitr) {
		goto label80
	}

	//     At this point, all the intervals have converged
	for i = savi1; i <= (*ilast); i++ {
		k = 2 * i
		ii = i - (*offset)
		//        All intervals marked by '0' have been refined.
		if (*iwork)[k-1-1] == 0 {
			w.Set(ii-1, half*(work.Get(k-1-1)+work.Get(k-1)))
			werr.Set(ii-1, work.Get(k-1)-w.Get(ii-1))
		}
	}
}
