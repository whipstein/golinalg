package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlarrb Given the relatively robust representation(RRR) L D L^T, DLARRB
// does "limited" bisection to refine the eigenvalues of L D L^T,
// W( IFIRST-OFFSET ) through W( ILAST-OFFSET ), to more accuracy. Initial
// guesses for these eigenvalues are input in W, the corresponding estimate
// of the error in these guesses and their gaps are input in WERR
// and WGAP, respectively. During bisection, intervals
// [left, right] are maintained by storing their mid-points and
// semi-widths in the arrays W and WERR respectively.
func Dlarrb(n *int, d, lld *mat.Vector, ifirst, ilast *int, rtol1, rtol2 *float64, offset *int, w, wgap, werr, work *mat.Vector, iwork *[]int, pivmin, spdiam *float64, twist, info *int) {
	var back, cvrgd, gap, half, left, lgap, mid, mnwdth, rgap, right, tmp, two, width, zero float64
	var i, i1, ii, ip, iter, k, maxitr, negcnt, next, nint, olnint, prev, r int

	zero = 0.0
	two = 2.0
	half = 0.5

	(*info) = 0

	//     Quick return if possible
	if (*n) <= 0 {
		return
	}

	maxitr = int((math.Log((*spdiam)+(*pivmin))-math.Log(*pivmin))/math.Log(two)) + 2
	mnwdth = two * (*pivmin)

	r = (*twist)
	if (r < 1) || (r > (*n)) {
		r = (*n)
	}

	//     Initialize unconverged intervals in [ WORK(2*I-1), WORK(2*I) ].
	//     The Sturm Count, Count( WORK(2*I-1) ) is arranged to be I-1, while
	//     Count( WORK(2*I) ) is stored in IWORK( 2*I ). The integer IWORK( 2*I-1 )
	//     for an unconverged interval is set to the index of the next unconverged
	//     interval, and is -1 or 0 for a converged interval. Thus a linked
	//     list of unconverged intervals is set up.
	i1 = (*ifirst)
	//     The number of unconverged intervals
	nint = 0
	//     The last unconverged interval found
	prev = 0
	rgap = wgap.Get(i1 - (*offset) - 1)
	for i = i1; i <= (*ilast); i++ {
		k = 2 * i
		ii = i - (*offset)
		left = w.Get(ii-1) - werr.Get(ii-1)
		right = w.Get(ii-1) + werr.Get(ii-1)
		lgap = rgap
		rgap = wgap.Get(ii - 1)
		gap = math.Min(lgap, rgap)
		//        Make sure that [LEFT,RIGHT] contains the desired eigenvalue
		//        Compute negcount from dstqds facto L+D+L+^T = L D L^T - LEFT
		//
		//        Do while( NEGCNT(LEFT).GT.I-1 )

		back = werr.Get(ii - 1)
	label20:
		;
		negcnt = Dlaneg(n, d, lld, &left, pivmin, &r)
		if negcnt > i-1 {
			left = left - back
			back = two * back
			goto label20
		}

		//        Do while( NEGCNT(RIGHT).LT.I )
		//        Compute negcount from dstqds facto L+D+L+^T = L D L^T - RIGHT
		back = werr.Get(ii - 1)
	label50:
		;
		negcnt = Dlaneg(n, d, lld, &right, pivmin, &r)
		if negcnt < i {
			right = right + back
			back = two * back
			goto label50
		}
		width = half * math.Abs(left-right)
		tmp = math.Max(math.Abs(left), math.Abs(right))
		cvrgd = math.Max((*rtol1)*gap, (*rtol2)*tmp)
		if width <= cvrgd || width <= mnwdth {
			//           This interval has already converged and does not need refinement.
			//           (Note that the gaps might change through refining the
			//            eigenvalues, however, they can only get bigger.)
			//           Remove it from the list.
			(*iwork)[k-1-1] = -1
			//           Make sure that I1 always points to the first unconverged interval
			if (i == i1) && (i < (*ilast)) {
				i1 = i + 1
			}
			if (prev >= i1) && (i <= (*ilast)) {
				(*iwork)[2*prev-1-1] = i + 1
			}
		} else {
			//           unconverged interval found
			prev = i
			nint = nint + 1
			(*iwork)[k-1-1] = i + 1
			(*iwork)[k-1] = negcnt
		}
		work.Set(k-1-1, left)
		work.Set(k-1, right)
	}

	//     Do while( NINT.GT.0 ), i.e. there are still unconverged intervals
	//     and while (ITER.LT.MAXITR)
	iter = 0
label80:
	;
	prev = i1 - 1
	i = i1
	olnint = nint
	for ip = 1; ip <= olnint; ip++ {
		k = 2 * i
		ii = i - (*offset)
		rgap = wgap.Get(ii - 1)
		lgap = rgap
		if ii > 1 {
			lgap = wgap.Get(ii - 1 - 1)
		}
		gap = math.Min(lgap, rgap)
		next = (*iwork)[k-1-1]
		left = work.Get(k - 1 - 1)
		right = work.Get(k - 1)
		mid = half * (left + right)
		//        semiwidth of interval
		width = right - mid
		tmp = math.Max(math.Abs(left), math.Abs(right))
		cvrgd = math.Max((*rtol1)*gap, (*rtol2)*tmp)
		if (width <= cvrgd) || (width <= mnwdth) || (iter == maxitr) {
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
		negcnt = Dlaneg(n, d, lld, &mid, pivmin, &r)
		if negcnt <= i-1 {
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
	for i = (*ifirst); i <= (*ilast); i++ {
		k = 2 * i
		ii = i - (*offset)
		//        All intervals marked by '0' have been refined.
		if (*iwork)[k-1-1] == 0 {
			w.Set(ii-1, half*(work.Get(k-1-1)+work.Get(k-1)))
			werr.Set(ii-1, work.Get(k-1)-w.Get(ii-1))
		}
	}

	for i = (*ifirst) + 1; i <= (*ilast); i++ {
		k = 2 * i
		ii = i - (*offset)
		wgap.Set(ii-1-1, math.Max(zero, w.Get(ii-1)-werr.Get(ii-1)-w.Get(ii-1-1)-werr.Get(ii-1-1)))
	}
}
