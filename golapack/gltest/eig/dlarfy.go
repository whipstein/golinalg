package eig

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
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
	var err error
	_ = err

	one = 1.0
	zero = 0.0
	half = 0.5

	if (*tau) == zero {
		return
	}

	//     Form  w:= C * v
	err = goblas.Dsymv(mat.UploByte(uplo), *n, one, c, v.Off(0, *incv), zero, work.Off(0, 1))

	alpha = -half * (*tau) * goblas.Ddot(*n, work.Off(0, 1), v.Off(0, *incv))
	goblas.Daxpy(*n, alpha, v.Off(0, *incv), work.Off(0, 1))

	//     C := C - v * w' - w * v'
	err = goblas.Dsyr2(mat.UploByte(uplo), *n, -(*tau), v.Off(0, *incv), work.Off(0, 1), c)
}
