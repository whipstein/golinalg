package golapack

import "golinalg/mat"

// Zlartv applies a vector of complex plane rotations with real cosines
// to elements of the complex vectors x and y. For i = 1,2,...,n
//
//    ( x(i) ) := (        c(i)   s(i) ) ( x(i) )
//    ( y(i) )    ( -conjg(s(i))  c(i) ) ( y(i) )
func Zlartv(n *int, x *mat.CVector, incx *int, y *mat.CVector, incy *int, c *mat.Vector, s *mat.CVector, incc *int) {
	var xi, yi complex128
	var i, ic, ix, iy int

	ix = 1
	iy = 1
	ic = 1
	for i = 1; i <= (*n); i++ {
		xi = x.Get(ix - 1)
		yi = y.Get(iy - 1)
		x.Set(ix-1, c.GetCmplx(ic-1)*xi+s.Get(ic-1)*yi)
		y.Set(iy-1, c.GetCmplx(ic-1)*yi-s.GetConj(ic-1)*xi)
		ix = ix + (*incx)
		iy = iy + (*incy)
		ic = ic + (*incc)
	}
}
