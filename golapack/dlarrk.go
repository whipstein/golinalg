package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlarrk computes one eigenvalue of a symmetric tridiagonal
// matrix T to suitable accuracy. This is an auxiliary code to be
// called from DSTEMR.
//
// To avoid overflow, the matrix must be scaled so that its
// largest element is no greater than overflow**(1/2) * underflow**(1/4) in absolute value, and for greatest
// accuracy, it should not be much smaller than that.
//
// See W. Kahan "Accurate Eigenvalues of a Symmetric Tridiagonal
// Matrix", Report CS41, Computer Science Dept., Stanford
// University, July 21, 1966.
func Dlarrk(n, iw int, gl, gu float64, d, e2 *mat.Vector, pivmin, reltol float64) (w, werr float64, info int) {
	var atoli, eps, fudge, half, left, mid, right, rtoli, tmp1, tmp2, tnorm, two, zero float64
	var i, it, itmax, negcnt int

	half = 0.5
	two = 2.0
	fudge = two
	zero = 0.0

	//     Quick return if possible
	if n <= 0 {
		info = 0
		return
	}

	//     Get machine constants
	eps = Dlamch(Precision)
	tnorm = math.Max(math.Abs(gl), math.Abs(gu))
	rtoli = reltol
	atoli = fudge * two * pivmin
	itmax = int((math.Log(tnorm+pivmin)-math.Log(pivmin))/math.Log(two)) + 2
	info = -1
	left = gl - fudge*tnorm*eps*float64(n) - fudge*two*pivmin
	right = gu + fudge*tnorm*eps*float64(n) + fudge*two*pivmin
	it = 0
label10:
	;

	//     Check if interval converged or maximum number of iterations reached
	tmp1 = math.Abs(right - left)
	tmp2 = math.Max(math.Abs(right), math.Abs(left))
	if tmp1 < math.Max(atoli, math.Max(pivmin, rtoli*tmp2)) {
		info = 0
		goto label30
	}
	if it > itmax {
		goto label30
	}

	//     Count number of negative pivots for mid-point
	it = it + 1
	mid = half * (left + right)
	negcnt = 0
	tmp1 = d.Get(0) - mid
	if math.Abs(tmp1) < pivmin {
		tmp1 = -pivmin
	}
	if tmp1 <= zero {
		negcnt = negcnt + 1
	}

	for i = 2; i <= n; i++ {
		tmp1 = d.Get(i-1) - e2.Get(i-1-1)/tmp1 - mid
		if math.Abs(tmp1) < pivmin {
			tmp1 = -pivmin
		}
		if tmp1 <= zero {
			negcnt = negcnt + 1
		}
	}
	if negcnt >= iw {
		right = mid
	} else {
		left = mid
	}
	goto label10
label30:
	;

	//     Converged or maximum number of iterations reached
	w = half * (left + right)
	werr = half * math.Abs(right-left)

	return
}
