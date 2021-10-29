package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dlarf applies a real elementary reflector H to a real m by n matrix
// C, from either the left or the right. H is represented in the form
//
//       H = I - tau * v * v**T
//
// where tau is a real scalar and v is a real vector.
//
// If tau = 0, then H is taken to be the unit matrix.
func Dlarf(side mat.MatSide, m, n int, v *mat.Vector, tau float64, c *mat.Matrix, work *mat.Vector) {
	var applyleft bool
	var one, zero float64
	var i, lastc, lastv int
	var err error

	one = 1.0
	zero = 0.0

	applyleft = side == Left
	lastv = 0
	lastc = 0
	if tau != zero {
		//!     Set up variables for scanning V.  LASTV begins pointing to the end
		//!     of V.
		if applyleft {
			lastv = m
		} else {
			lastv = n
		}
		if v.Inc > 0 {
			i = 1 + (lastv-1)*v.Inc
		} else {
			i = 1
		}
		//!     Look for the last non-zero row in V.

		for lastv > 0 && v.Get(i-1) == zero {
			lastv = lastv - 1
			i = i - v.Inc
		}
		if applyleft {
			//!     Scan for the last non-zero column in C(1:lastv,:).

			lastc = Iladlc(lastv, n, c)
		} else {
			//!     Scan for the last non-zero row in C(:,1:lastv).

			lastc = Iladlr(m, lastv, c)
		}
	}
	//!     Note that lastc.eq.0 renders the BLAS operations null; no special
	//!     case is needed at this level.
	if applyleft {
		//        Form  H * C
		if lastv > 0 {
			//           w(1:lastc,1) := C(1:lastv,1:lastc)**T * v(1:lastv,1)
			if err = goblas.Dgemv(Trans, lastv, lastc, one, c, v, zero, work.Off(0, 1)); err != nil {
				panic(err)
			}

			//           C(1:lastv,1:lastc) := C(...) - v(1:lastv,1) * w(1:lastc,1)**T
			if err = goblas.Dger(lastv, lastc, -tau, v, work.Off(0, 1), c); err != nil {
				panic(err)
			}
		}
	} else {
		//        Form  C * H
		if lastv > 0 {
			//           w(1:lastc,1) := C(1:lastc,1:lastv) * v(1:lastv,1)
			if err = goblas.Dgemv(NoTrans, lastc, lastv, one, c, v, zero, work.Off(0, 1)); err != nil {
				panic(err)
			}

			//           C(1:lastc,1:lastv) := C(...) - w(1:lastc,1) * v(1:lastv,1)**T
			if err = goblas.Dger(lastc, lastv, -tau, work.Off(0, 1), v, c); err != nil {
				panic(err)
			}
		}
	}
}
