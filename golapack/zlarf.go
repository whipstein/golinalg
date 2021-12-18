package golapack

import (
	"github.com/whipstein/golinalg/mat"
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
func Zlarf(side mat.MatSide, m, n int, v *mat.CVector, incv int, tau complex128, c *mat.CMatrix, work *mat.CVector) {
	var applyleft bool
	var one, zero complex128
	var i, lastc, lastv int
	var err error

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	applyleft = side == Left
	lastv = 0
	lastc = 0
	if tau != zero {
		//     Set up variables for scanning V.  LASTV begins pointing to the end
		//     of V.
		if applyleft {
			lastv = m
		} else {
			lastv = n
		}
		if incv > 0 {
			i = 1 + (lastv-1)*incv
		} else {
			i = 1
		}
		//     Look for the last non-zero row in V.
		for lastv > 0 && v.Get(i-1) == zero {
			lastv = lastv - 1
			i = i - incv
		}
		if applyleft {
			//     Scan for the last non-zero column in C(1:lastv,:).
			lastc = Ilazlc(lastv, n, c)
		} else {
			//     Scan for the last non-zero row in C(:,1:lastv).
			lastc = Ilazlr(m, lastv, c)
		}
	}
	//     Note that lastc.eq.0 renders the BLAS operations null; no special
	//     case is needed at this level.
	if applyleft {
		//        Form  H * C
		if lastv > 0 {
			//           w(1:lastc,1) := C(1:lastv,1:lastc)**H * v(1:lastv,1)
			if err = work.Gemv(ConjTrans, lastv, lastc, one, c, v, incv, zero, 1); err != nil {
				panic(err)
			}

			//           C(1:lastv,1:lastc) := C(...) - v(1:lastv,1) * w(1:lastc,1)**H
			if err = c.Gerc(lastv, lastc, -tau, v, incv, work, 1); err != nil {
				panic(err)
			}
		}
	} else {
		//        Form  C * H
		if lastv > 0 {
			//           w(1:lastc,1) := C(1:lastc,1:lastv) * v(1:lastv,1)
			if err = work.Gemv(NoTrans, lastc, lastv, one, c, v, incv, zero, 1); err != nil {
				panic(err)
			}

			//           C(1:lastc,1:lastv) := C(...) - w(1:lastc,1) * v(1:lastv,1)**H
			if err = c.Gerc(lastc, lastv, -tau, work, 1, v, incv); err != nil {
				panic(err)
			}
		}
	}
}
