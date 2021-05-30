package golapack

import (
	"golinalg/goblas"
	"golinalg/mat"
)

// Zlarf applies a complex elementary reflector H to a complex M-by-N
// matrix C, from either the left or the right. H is represented in the
// form
//
//       H = I - tau * v * v**H
//
// where tau is a complex scalar and v is a complex vector.
//
// If tau = 0, then H is taken to be the unit matrix.
//
// To apply H**H, supply conjg(tau) instead
// tau.
func Zlarf(side byte, m, n *int, v *mat.CVector, incv *int, tau *complex128, c *mat.CMatrix, ldc *int, work *mat.CVector) {
	var applyleft bool
	var one, zero complex128
	var i, lastc, lastv int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	applyleft = side == 'L'
	lastv = 0
	lastc = 0
	if (*tau) != zero {
		//     Set up variables for scanning V.  LASTV begins pointing to the end
		//     of V.
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
		//     Look for the last non-zero row in V.
		for lastv > 0 && v.Get(i-1) == zero {
			lastv = lastv - 1
			i = i - (*incv)
		}
		if applyleft {
			//     Scan for the last non-zero column in C(1:lastv,:).
			lastc = Ilazlc(&lastv, n, c, ldc)
		} else {
			//     Scan for the last non-zero row in C(:,1:lastv).
			lastc = Ilazlr(m, &lastv, c, ldc)
		}
	}
	//     Note that lastc.eq.0 renders the BLAS operations null; no special
	//     case is needed at this level.
	if applyleft {
		//        Form  H * C
		if lastv > 0 {
			//           w(1:lastc,1) := C(1:lastv,1:lastc)**H * v(1:lastv,1)
			goblas.Zgemv(ConjTrans, &lastv, &lastc, &one, c, ldc, v, incv, &zero, work, func() *int { y := 1; return &y }())

			//           C(1:lastv,1:lastc) := C(...) - v(1:lastv,1) * w(1:lastc,1)**H
			goblas.Zgerc(&lastv, &lastc, toPtrc128(-(*tau)), v, incv, work, func() *int { y := 1; return &y }(), c, ldc)
		}
	} else {
		//        Form  C * H
		if lastv > 0 {
			//           w(1:lastc,1) := C(1:lastc,1:lastv) * v(1:lastv,1)
			goblas.Zgemv(NoTrans, &lastc, &lastv, &one, c, ldc, v, incv, &zero, work, func() *int { y := 1; return &y }())

			//           C(1:lastc,1:lastv) := C(...) - w(1:lastc,1) * v(1:lastv,1)**H
			goblas.Zgerc(&lastc, &lastv, toPtrc128(-(*tau)), work, func() *int { y := 1; return &y }(), v, incv, c, ldc)
		}
	}
}
