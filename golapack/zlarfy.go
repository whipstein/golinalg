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
	var err error
	_ = err

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)
	half = (0.5 + 0.0*1i)

	if (*tau) == zero {
		return
	}

	//     Form  w:= C * v
	err = goblas.Zhemv(mat.UploByte(uplo), *n, one, c, *ldc, v, *incv, zero, work, 1)

	alpha = -half * (*tau) * goblas.Zdotc(*n, work, 1, v, *incv)
	goblas.Zaxpy(*n, alpha, v, *incv, work, 1)

	//     C := C - v * w' - w * v'
	err = goblas.Zher2(mat.UploByte(uplo), *n, -(*tau), v, *incv, work, 1, c, *ldc)
}
