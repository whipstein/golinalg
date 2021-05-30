package golapack

import (
	"golinalg/mat"
	"math"
)

// Dlargv generates a vector of real plane rotations, determined by
// elements of the real vectors x and y. For i = 1,2,...,n
//
//    (  c(i)  s(i) ) ( x(i) ) = ( a(i) )
//    ( -s(i)  c(i) ) ( y(i) ) = (   0  )
func Dlargv(n *int, x *mat.Vector, incx *int, y *mat.Vector, incy *int, c *mat.Vector, incc *int) {
	var f, g, one, t, tt, zero float64
	var i, ic, ix, iy int

	zero = 0.0
	one = 1.0

	ix = 1
	iy = 1
	ic = 1
	for i = 1; i <= (*n); i++ {
		f = x.Get(ix - 1)
		g = y.Get(iy - 1)
		if g == zero {
			c.Set(ic-1, one)
		} else if f == zero {
			c.Set(ic-1, zero)
			y.Set(iy-1, one)
			x.Set(ix-1, g)
		} else if math.Abs(f) > math.Abs(g) {
			t = g / f
			tt = math.Sqrt(one + t*t)
			c.Set(ic-1, one/tt)
			y.Set(iy-1, t*c.Get(ic-1))
			x.Set(ix-1, f*tt)
		} else {
			t = f / g
			tt = math.Sqrt(one + t*t)
			y.Set(iy-1, one/tt)
			c.Set(ic-1, t*y.Get(iy-1))
			x.Set(ix-1, g*tt)
		}
		ic = ic + (*incc)
		iy = iy + (*incy)
		ix = ix + (*incx)
	}
}
