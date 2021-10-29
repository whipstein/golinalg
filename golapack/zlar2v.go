package golapack

import (
	"math/cmplx"

	"github.com/whipstein/golinalg/mat"
)

// Zlar2v applies a vector of complex plane rotations with real cosines
// from both sides to a sequence of 2-by-2 complex Hermitian matrices,
// defined by the elements of the vectors x, y and z. For i = 1,2,...,n
//
//    (       x(i)  z(i) ) :=
//    ( conjg(z(i)) y(i) )
//
//      (  c(i) conjg(s(i)) ) (       x(i)  z(i) ) ( c(i) -conjg(s(i)) )
//      ( -s(i)       c(i)  ) ( conjg(z(i)) y(i) ) ( s(i)        c(i)  )
func Zlar2v(n int, x, y, z *mat.CVector, c *mat.Vector, s *mat.CVector) {
	var si, t2, t3, t4, zi complex128
	var ci, sii, sir, t1i, t1r, t5, t6, xi, yi, zii, zir float64
	var i, ic, ix int

	ix = 1
	ic = 1
	for i = 1; i <= n; i++ {
		xi = x.GetRe(ix - 1)
		yi = y.GetRe(ix - 1)
		zi = z.Get(ix - 1)
		zir = real(zi)
		zii = imag(zi)
		ci = c.Get(ic - 1)
		si = s.Get(ic - 1)
		sir = real(si)
		sii = imag(si)
		t1r = sir*zir - sii*zii
		t1i = sir*zii + sii*zir
		t2 = complex(ci, 0) * zi
		t3 = t2 - cmplx.Conj(si)*complex(xi, 0)
		t4 = cmplx.Conj(t2) + si*complex(yi, 0)
		t5 = ci*xi + t1r
		t6 = ci*yi - t1r
		x.SetRe(ix-1, ci*t5+(sir*real(t4)+sii*imag(t4)))
		y.SetRe(ix-1, ci*t6-(sir*real(t3)-sii*imag(t3)))
		z.Set(ix-1, complex(ci, 0)*t3+cmplx.Conj(si)*complex(t6, t1i))
		ix = ix + x.Inc
		ic = ic + c.Inc
	}
}
