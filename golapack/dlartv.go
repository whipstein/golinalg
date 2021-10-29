package golapack

import "github.com/whipstein/golinalg/mat"

// Dlartv applies a vector of real plane rotations to elements of the
// real vectors x and y. For i = 1,2,...,n
//
//    ( x(i) ) := (  c(i)  s(i) ) ( x(i) )
//    ( y(i) )    ( -s(i)  c(i) ) ( y(i) )
func Dlartv(n int, x, y, c, s *mat.Vector) {
	var xi, yi float64
	var i, ic, ix, iy int

	ix = 1
	iy = 1
	ic = 1
	for i = 1; i <= n; i++ {
		xi = x.Get(ix - 1)
		yi = y.Get(iy - 1)
		x.Set(ix-1, c.Get(ic-1)*xi+s.Get(ic-1)*yi)
		y.Set(iy-1, c.Get(ic-1)*yi-s.Get(ic-1)*xi)
		ix = ix + x.Inc
		iy = iy + y.Inc
		ic = ic + c.Inc
	}
}
