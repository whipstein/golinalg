package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dstect counts the number NUM of eigenvalues of a tridiagonal
//    matrix T which are less than or equal to SHIFT. T has
//    diagonal entries A(1), ... , A(N), and offdiagonal entries
//    B(1), ..., B(N-1).
//    See W. Kahan "Accurate Eigenvalues of a Symmetric Tridiagonal
//    Matrix", Report CS41, Computer Science Dept., Stanford
//    University, July 21, 1966
func Dstect(n *int, a, b *mat.Vector, shift *float64, num *int) {
	var m1, m2, mx, one, ovfl, sov, sshift, ssun, sun, three, tmp, tom, u, unfl, zero float64
	var i int

	zero = 0.0
	one = 1.0
	three = 3.0

	//     Get machine constants
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = golapack.Dlamch(Overflow)

	//     Find largest entry
	mx = math.Abs(a.Get(0))
	for i = 1; i <= (*n)-1; i++ {
		mx = math.Max(mx, math.Max(math.Abs(a.Get(i)), math.Abs(b.Get(i-1))))
	}

	//     Handle easy cases, including zero matrix
	if (*shift) >= three*mx {
		(*num) = (*n)
		return
	}
	if (*shift) < -three*mx {
		(*num) = 0
		return
	}

	//     Compute scale factors as in Kahan's report
	//     At this point, MX .NE. 0 so we can divide by it
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
	(*num) = 0
	sshift = ((*shift) * m1) * m2
	u = (a.Get(0)*m1)*m2 - sshift
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
	for i = 2; i <= (*n); i++ {
		tmp = (b.Get(i-1-1) * m1) * m2
		u = ((a.Get(i-1)*m1)*m2 - tmp*(tmp/u)) - sshift
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
