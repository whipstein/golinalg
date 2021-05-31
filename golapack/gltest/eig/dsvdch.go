package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dsvdch checks to see if SVD(1) ,..., SVD(N) are accurate singular
// values of the bidiagonal matrix B with diagonal entries
// S(1) ,..., S(N) and superdiagonal entries E(1) ,..., E(N-1)).
// It does this by expanding each SVD(I) into an interval
// [SVD(I) * (1-EPS) , SVD(I) * (1+EPS)], merging overlapping intervals
// if any, and using Sturm sequences to count and verify whether each
// resulting interval has the correct number of singular values (using
// DSVDCT). Here EPS=TOL*MAX(N/10,1)*MAZHEP, where MACHEP is the
// machine precision. The routine assumes the singular values are sorted
// with SVD(1) the largest and SVD(N) smallest.  If each interval
// contains the correct number of singular values, INFO = 0 is returned,
// otherwise INFO is the index of the first singular value in the first
// bad interval.
func Dsvdch(n *int, s, e, svd *mat.Vector, tol *float64, info *int) {
	var eps, lower, one, ovfl, tuppr, unfl, unflep, upper, zero float64
	var bpnt, count, numl, numu, tpnt int

	one = 1.0
	zero = 0.0

	//     Get machine constants
	(*info) = 0
	if (*n) <= 0 {
		return
	}
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = golapack.Dlamch(Overflow)
	eps = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)

	//     UNFLEP is chosen so that when an eigenvalue is multiplied by the
	//     scale factor sqrt(OVFL)*sqrt(sqrt(UNFL))/MX in DSVDCT, it exceeds
	//     sqrt(UNFL), which is the lower limit for DSVDCT.
	unflep = (math.Sqrt(math.Sqrt(unfl))/math.Sqrt(ovfl))*svd.Get(0) + unfl/eps

	//     The value of EPS works best when TOL .GE. 10.
	eps = (*tol) * float64(maxint((*n)/10, 1)) * eps

	//     TPNT points to singular value at right endpoint of interval
	//     BPNT points to singular value at left  endpoint of interval
	tpnt = 1
	bpnt = 1

	//     Begin loop over all intervals
label10:
	;
	upper = (one+eps)*svd.Get(tpnt-1) + unflep
	lower = (one-eps)*svd.Get(bpnt-1) - unflep
	if lower <= unflep {
		lower = -upper
	}

	//     Begin loop merging overlapping intervals
label20:
	;
	if bpnt == (*n) {
		goto label30
	}
	tuppr = (one+eps)*svd.Get(bpnt+1-1) + unflep
	if tuppr < lower {
		goto label30
	}

	//     Merge
	bpnt = bpnt + 1
	lower = (one-eps)*svd.Get(bpnt-1) - unflep
	if lower <= unflep {
		lower = -upper
	}
	goto label20
label30:
	;

	//     Count singular values in interval [ LOWER, UPPER ]
	Dsvdct(n, s, e, &lower, &numl)
	Dsvdct(n, s, e, &upper, &numu)
	count = numu - numl
	if lower < zero {
		count = count / 2
	}
	if count != bpnt-tpnt+1 {
		//        Wrong number of singular values in interval
		(*info) = tpnt
		return
	}
	tpnt = bpnt + 1
	bpnt = tpnt
	if tpnt <= (*n) {
		goto label10
	}
}
