package eig

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// dlarfy applies an elementary reflector, or Householder matrix, H,
// to an n x n symmetric matrix C, from both the left and the right.
//
// H is represented in the form
//
//    H = I - tau * v * v'
//
// where  tau  is a scalar and  v  is a vector.
//
// If  tau  is  zero, then  H  is taken to be the unit matrix.
func dlarfy(uplo mat.MatUplo, n int, v *mat.Vector, tau float64, c *mat.Matrix, work *mat.Vector) {
	var alpha, half, one, zero float64
	var err error

	one = 1.0
	zero = 0.0
	half = 0.5

	if tau == zero {
		return
	}

	//     Form  w:= C * v
	if err = goblas.Dsymv(uplo, n, one, c, v, zero, work.Off(0, 1)); err != nil {
		panic(err)
	}

	alpha = -half * tau * goblas.Ddot(n, work.Off(0, 1), v)
	goblas.Daxpy(n, alpha, v, work.Off(0, 1))

	//     C := C - v * w' - w * v'
	if err = goblas.Dsyr2(uplo, n, -tau, v, work.Off(0, 1), c); err != nil {
		panic(err)
	}
}
