package golapack

import (
	"golinalg/goblas"
	"golinalg/mat"
)

// Dlarf applies a real elementary reflector H to a real m by n matrix
// C, from either the left or the right. H is represented in the form
//
//       H = I - tau * v * v**T
//
// where tau is a real scalar and v is a real vector.
//
// If tau = 0, then H is taken to be the unit matrix.
func Dlarf(side byte, m, n *int, v *mat.Vector, incv *int, tau *float64, c *mat.Matrix, ldc *int, work *mat.Vector) {
	var applyleft bool
	var one, zero float64
	var i, lastc, lastv int

	one = 1.0
	zero = 0.0

	applyleft = side == 'L'
	lastv = 0
	lastc = 0
	if (*tau) != zero {
		//!     Set up variables for scanning V.  LASTV begins pointing to the end
		//!     of V.
		if applyleft {
			lastv = (*m)
		} else {
			lastv = (*n)
		}
		if (*incv) > 0 {
			i = 1 + (lastv-1)*(*incv)
		} else {
			i = 1
		}
		//!     Look for the last non-zero row in V.

		for lastv > 0 && v.Get(i-1) == zero {
			lastv = lastv - 1
			i = i - (*incv)
		}
		if applyleft {
			//!     Scan for the last non-zero column in C(1:lastv,:).

			lastc = Iladlc(&lastv, n, c, ldc)
		} else {
			//!     Scan for the last non-zero row in C(:,1:lastv).

			lastc = Iladlr(m, &lastv, c, ldc)
		}
	}
	//!     Note that lastc.eq.0 renders the BLAS operations null; no special
	//!     case is needed at this level.
	if applyleft {
		//        Form  H * C
		if lastv > 0 {
			//           w(1:lastc,1) := C(1:lastv,1:lastc)**T * v(1:lastv,1)
			goblas.Dgemv(Trans, &lastv, &lastc, &one, c, ldc, v, incv, &zero, work, func() *int { y := 1; return &y }())

			//           C(1:lastv,1:lastc) := C(...) - v(1:lastv,1) * w(1:lastc,1)**T
			goblas.Dger(&lastv, &lastc, toPtrf64(-(*tau)), v, incv, work, func() *int { y := 1; return &y }(), c, ldc)
		}
	} else {
		//        Form  C * H
		if lastv > 0 {
			//           w(1:lastc,1) := C(1:lastc,1:lastv) * v(1:lastv,1)
			goblas.Dgemv(NoTrans, &lastc, &lastv, &one, c, ldc, v, incv, &zero, work, func() *int { y := 1; return &y }())

			//           C(1:lastc,1:lastv) := C(...) - w(1:lastc,1) * v(1:lastv,1)**T
			goblas.Dger(&lastc, &lastv, toPtrf64(-(*tau)), work, func() *int { y := 1; return &y }(), v, incv, c, ldc)
		}
	}
}
