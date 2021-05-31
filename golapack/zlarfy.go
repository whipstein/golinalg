package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Zlarfy applies an elementary reflector, or Householder matrix, H,
// to an n x n Hermitian matrix C, from both the left and the right.
//
// H is represented in the form
//
//    H = I - tau * v * v'
//
// where  tau  is a scalar and  v  is a vector.
//
// If  tau  is  zero, then  H  is taken to be the unit matrix.
func Zlarfy(uplo byte, n *int, v *mat.CVector, incv *int, tau *complex128, c *mat.CMatrix, ldc *int, work *mat.CVector) {
	var alpha, half, one, zero complex128

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)
	half = (0.5 + 0.0*1i)

	if (*tau) == zero {
		return
	}

	//     Form  w:= C * v
	goblas.Zhemv(mat.UploByte(uplo), n, &one, c, ldc, v, incv, &zero, work, func() *int { y := 1; return &y }())

	alpha = -half * (*tau) * goblas.Zdotc(n, work, func() *int { y := 1; return &y }(), v, incv)
	goblas.Zaxpy(n, &alpha, v, incv, work, func() *int { y := 1; return &y }())

	//     C := C - v * w' - w * v'
	goblas.Zher2(mat.UploByte(uplo), n, toPtrc128(-(*tau)), v, incv, work, func() *int { y := 1; return &y }(), c, ldc)
}
