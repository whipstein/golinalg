package golapack

import "github.com/whipstein/golinalg/mat"

// Dlar2v applies a vector of real plane rotations from both sides to
// a sequence of 2-by-2 real symmetric matrices, defined by the elements
// of the vectors x, y and z. For i = 1,2,...,n
//
//    ( x(i)  z(i) ) := (  c(i)  s(i) ) ( x(i)  z(i) ) ( c(i) -s(i) )
//    ( z(i)  y(i) )    ( -s(i)  c(i) ) ( z(i)  y(i) ) ( s(i)  c(i) )
func Dlar2v(n *int, x, y, z *mat.Vector, incx *int, c, s *mat.Vector, incc *int) {
	var ci, si, t1, t2, t3, t4, t5, t6, xi, yi, zi float64
	var i, ic, ix int

	ix = 1
	ic = 1
	for i = 1; i <= (*n); i++ {
		xi = x.Get(ix - 1)
		yi = y.Get(ix - 1)
		zi = z.Get(ix - 1)
		ci = c.Get(ic - 1)
		si = s.Get(ic - 1)
		t1 = si * zi
		t2 = ci * zi
		t3 = t2 - si*xi
		t4 = t2 + si*yi
		t5 = ci*xi + t1
		t6 = ci*yi - t1
		x.Set(ix-1, ci*t5+si*t4)
		y.Set(ix-1, ci*t6-si*t3)
		z.Set(ix-1, ci*t4-si*t5)
		ix = ix + (*incx)
		ic = ic + (*incc)
	}
}
