package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlaqr1 Given a 2-by-2 or 3-by-3 matrix H, DLAQR1 sets v to a
//      scalar multiple of the first column of the product
//
//      (*)  K = (H - (sr1 + i*si1)*I)*(H - (sr2 + i*si2)*I)
//
//      scaling to avoid overflows and most underflows. It
//      is assumed that either
//
//              1) sr1 = sr2 and si1 = -si2
//          or
//              2) si1 = si2 = 0.
//
//      This is useful for starting double implicit shift bulges
//      in the QR algorithm.
func Dlaqr1(n int, h *mat.Matrix, sr1, si1, sr2, si2 float64, v *mat.Vector) {
	var h21s, h31s, s, zero float64

	zero = 0.0

	//     Quick return if possible
	if n != 2 && n != 3 {
		return
	}

	if n == 2 {
		s = math.Abs(h.Get(0, 0)-sr2) + math.Abs(si2) + math.Abs(h.Get(1, 0))
		if s == zero {
			v.Set(0, zero)
			v.Set(1, zero)
		} else {
			h21s = h.Get(1, 0) / s
			v.Set(0, h21s*h.Get(0, 1)+(h.Get(0, 0)-sr1)*((h.Get(0, 0)-sr2)/s)-si1*(si2/s))
			v.Set(1, h21s*(h.Get(0, 0)+h.Get(1, 1)-sr1-sr2))
		}
	} else {
		s = math.Abs(h.Get(0, 0)-sr2) + math.Abs(si2) + math.Abs(h.Get(1, 0)) + math.Abs(h.Get(2, 0))
		if s == zero {
			v.Set(0, zero)
			v.Set(1, zero)
			v.Set(2, zero)
		} else {
			h21s = h.Get(1, 0) / s
			h31s = h.Get(2, 0) / s
			v.Set(0, (h.Get(0, 0)-sr1)*((h.Get(0, 0)-sr2)/s)-si1*(si2/s)+h.Get(0, 1)*h21s+h.Get(0, 2)*h31s)
			v.Set(1, h21s*(h.Get(0, 0)+h.Get(1, 1)-sr1-sr2)+h.Get(1, 2)*h31s)
			v.Set(2, h31s*(h.Get(0, 0)+h.Get(2, 2)-sr1-sr2)+h21s*h.Get(2, 1))
		}
	}
}
