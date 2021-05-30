package eig

import (
	"golinalg/goblas"
	"golinalg/mat"
)

// Dlarfy applies an elementary reflector, or Householder matrix, H,
// to an n x n symmetric matrix C, from both the left and the right.
//
// H is represented in the form
//
//    H = I - tau * v * v'
//
// where  tau  is a scalar and  v  is a vector.
//
// If  tau  is  zero, then  H  is taken to be the unit matrix.
func Dlarfy(uplo byte, n *int, v *mat.Vector, incv *int, tau *float64, c *mat.Matrix, ldc *int, work *mat.Vector) {
	var alpha, half, one, zero float64

	one = 1.0
	zero = 0.0
	half = 0.5

	if (*tau) == zero {
		return
	}

	//     Form  w:= C * v
	goblas.Dsymv(mat.UploByte(uplo), n, &one, c, ldc, v, incv, &zero, work, func() *int { y := 1; return &y }())

	alpha = -half * (*tau) * goblas.Ddot(n, work, func() *int { y := 1; return &y }(), v, incv)
	goblas.Daxpy(n, &alpha, v, incv, work, func() *int { y := 1; return &y }())

	//     C := C - v * w' - w * v'
	goblas.Dsyr2(mat.UploByte(uplo), n, toPtrf64(-(*tau)), v, incv, work, func() *int { y := 1; return &y }(), c, ldc)
}
