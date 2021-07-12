package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dsvdct counts the number NUM of eigenvalues of a 2*N by 2*N
// tridiagonal matrix T which are less than or equal to SHIFT.  T is
// formed by putting zeros on the diagonal and making the off-diagonals
// equal to S(1), E(1), S(2), E(2), ... , E(N-1), S(N).  If SHIFT is
// positive, NUM is equal to N plus the number of singular values of a
// bidiagonal matrix B less than or equal to SHIFT.  Here B has diagonal
// entries S(1), ..., S(N) and superdiagonal entries E(1), ... E(N-1).
// If SHIFT is negative, NUM is equal to the number of singular values
// of B greater than or equal to -SHIFT.
//
// See W. Kahan "Accurate Eigenvalues of a Symmetric Tridiagonal
// Matrix", Report CS41, Computer Science Dept., Stanford University,
// July 21, 1966
func Dsvdct(n *int, s, e *mat.Vector, shift *float64, num *int) {
	var m1, m2, mx, one, ovfl, sov, sshift, ssun, sun, tmp, tom, u, unfl, zero float64
	var i int

	one = 1.0
	zero = 0.0

	//     Get machine constants
	unfl = 2 * golapack.Dlamch(SafeMinimum)
	ovfl = one / unfl

	//     Find largest entry
	mx = s.GetMag(0)
	for i = 1; i <= (*n)-1; i++ {
		mx = math.Max(mx, math.Max(s.GetMag(i), e.GetMag(i-1)))
	}

	if mx == zero {
		if (*shift) < zero {
			(*num) = 0
		} else {
			(*num) = 2 * (*n)
		}
		return
	}

	//     Compute scale factors as in Kahan's report
	sun = math.Sqrt(unfl)
	ssun = math.Sqrt(sun)
	sov = math.Sqrt(ovfl)
	tom = ssun * sov
	if mx <= one {
		m1 = one / mx
		m2 = tom
	} else {
		m1 = one
		m2 = tom / mx
	}

	//     Begin counting
	u = one
	(*num) = 0
	sshift = ((*shift) * m1) * m2
	u = -sshift
	if u <= sun {
		if u <= zero {
			(*num) = (*num) + 1
			if u > -sun {
				u = -sun
			}
		} else {
			u = sun
		}
	}
	tmp = (s.Get(0) * m1) * m2
	u = -tmp*(tmp/u) - sshift
	if u <= sun {
		if u <= zero {
			(*num) = (*num) + 1
			if u > -sun {
				u = -sun
			}
		} else {
			u = sun
		}
	}
	for i = 1; i <= (*n)-1; i++ {
		tmp = (e.Get(i-1) * m1) * m2
		u = -tmp*(tmp/u) - sshift
		if u <= sun {
			if u <= zero {
				(*num) = (*num) + 1
				if u > -sun {
					u = -sun
				}
			} else {
				u = sun
			}
		}
		tmp = (s.Get(i) * m1) * m2
		u = -tmp*(tmp/u) - sshift
		if u <= sun {
			if u <= zero {
				(*num) = (*num) + 1
				if u > -sun {
					u = -sun
				}
			} else {
				u = sun
			}
		}
	}
}
