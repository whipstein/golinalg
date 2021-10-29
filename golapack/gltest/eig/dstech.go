package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dstech Let T be the tridiagonal matrix with diagonal entries A(1) ,...,
//    A(N) and offdiagonal entries B(1) ,..., B(N-1)).  DSTECH checks to
//    see if EIG(1) ,..., EIG(N) are indeed accurate eigenvalues of T.
//    It does this by expanding each EIG(I) into an interval
//    [SVD(I) - EPS, SVD(I) + EPS], merging overlapping intervals if
//    any, and using Sturm sequences to count and verify whether each
//    resulting interval has the correct number of eigenvalues (using
//    DSTECT).  Here EPS = TOL*MAZHEPS*MAXEIG, where MACHEPS is the
//    machine precision and MAXEIG is the absolute value of the largest
//    eigenvalue. If each interval contains the correct number of
//    eigenvalues, INFO = 0 is returned, otherwise INFO is the index of
//    the first eigenvalue in the first bad interval.
func dstech(n int, a, b, eig *mat.Vector, tol float64, work *mat.Vector) (info int) {
	var emin, eps, lower, mx, tuppr, unflep, upper, zero float64
	var bpnt, count, i, isub, j, numl, numu, tpnt int

	zero = 0.0

	//     Check input parameters
	info = 0
	if n == 0 {
		return
	}
	if n < 0 {
		info = -1
		return
	}
	if tol < zero {
		info = -5
		return
	}

	//     Get machine constants
	eps = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)
	unflep = golapack.Dlamch(SafeMinimum) / eps
	eps = tol * eps

	//     Compute maximum absolute eigenvalue, error tolerance
	mx = math.Abs(eig.Get(0))
	for i = 2; i <= n; i++ {
		mx = math.Max(mx, math.Abs(eig.Get(i-1)))
	}
	eps = math.Max(eps*mx, unflep)

	//     Sort eigenvalues from EIG into WORK
	for i = 1; i <= n; i++ {
		work.Set(i-1, eig.Get(i-1))
	}
	for i = 1; i <= n-1; i++ {
		isub = 1
		emin = work.Get(0)
		for j = 2; j <= n+1-i; j++ {
			if work.Get(j-1) < emin {
				isub = j
				emin = work.Get(j - 1)
			}
		}
		if isub != n+1-i {
			work.Set(isub-1, work.Get(n+1-i-1))
			work.Set(n+1-i-1, emin)
		}
	}

	//     TPNT points to singular value at right endpoint of interval
	//     BPNT points to singular value at left  endpoint of interval
	tpnt = 1
	bpnt = 1

	//     Begin loop over all intervals
label50:
	;
	upper = work.Get(tpnt-1) + eps
	lower = work.Get(bpnt-1) - eps

	//     Begin loop merging overlapping intervals
label60:
	;
	if bpnt == n {
		goto label70
	}
	tuppr = work.Get(bpnt) + eps
	if tuppr < lower {
		goto label70
	}

	//     Merge
	bpnt = bpnt + 1
	lower = work.Get(bpnt-1) - eps
	goto label60
label70:
	;

	//     Count singular values in interval [ LOWER, UPPER ]
	numl = dstect(n, a, b, lower)
	numu = dstect(n, a, b, upper)
	count = numu - numl
	if count != bpnt-tpnt+1 {
		//        Wrong number of singular values in interval
		info = tpnt
		return
	}
	tpnt = bpnt + 1
	bpnt = tpnt
	if tpnt <= n {
		goto label50
	}

	return
}
