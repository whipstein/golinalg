package golapack

import "github.com/whipstein/golinalg/mat"

// Zlaqr1 Given a 2-by-2 or 3-by-3 matrix H, ZLAQR1 sets v to a
//      scalar multiple of the first column of the product
//
//      (*)  K = (H - s1*I)*(H - s2*I)
//
//      scaling to avoid overflows and most underflows.
//
//      This is useful for starting double implicit shift bulges
//      in the QR algorithm.
func Zlaqr1(n int, h *mat.CMatrix, s1, s2 complex128, v *mat.CVector) {
	var h21s, h31s, zero complex128
	var rzero, s float64

	zero = (0.0 + 0.0*1i)
	rzero = 0.0

	//     Quick return if possible
	if n != 2 && n != 3 {
		return
	}

	if n == 2 {
		s = cabs1(h.Get(0, 0)-s2) + cabs1(h.Get(1, 0))
		if s == rzero {
			v.Set(0, zero)
			v.Set(1, zero)
		} else {
			h21s = h.Get(1, 0) / complex(s, 0)
			v.Set(0, h21s*h.Get(0, 1)+(h.Get(0, 0)-s1)*((h.Get(0, 0)-s2)/complex(s, 0)))
			v.Set(1, h21s*(h.Get(0, 0)+h.Get(1, 1)-s1-s2))
		}
	} else {
		s = cabs1(h.Get(0, 0)-s2) + cabs1(h.Get(1, 0)) + cabs1(h.Get(2, 0))
		if complex(s, 0) == zero {
			v.Set(0, zero)
			v.Set(1, zero)
			v.Set(2, zero)
		} else {
			h21s = h.Get(1, 0) / complex(s, 0)
			h31s = h.Get(2, 0) / complex(s, 0)
			v.Set(0, (h.Get(0, 0)-s1)*((h.Get(0, 0)-s2)/complex(s, 0))+h.Get(0, 1)*h21s+h.Get(0, 2)*h31s)
			v.Set(1, h21s*(h.Get(0, 0)+h.Get(1, 1)-s1-s2)+h.Get(1, 2)*h31s)
			v.Set(2, h31s*(h.Get(0, 0)+h.Get(2, 2)-s1-s2)+h21s*h.Get(2, 1))
		}
	}
}
