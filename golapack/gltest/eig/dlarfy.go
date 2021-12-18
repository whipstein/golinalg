package eig

import (
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
func dlarfy(uplo mat.MatUplo, n int, v *mat.Vector, incv int, tau float64, c *mat.Matrix, work *mat.Vector) {
	var alpha, half, one, zero float64
	var err error

	one = 1.0
	zero = 0.0
	half = 0.5

	if tau == zero {
		return
	}

	//     Form  w:= C * v
	if err = work.Symv(uplo, n, one, c, v, incv, zero, 1); err != nil {
		panic(err)
	}

	alpha = -half * tau * v.Dot(n, work, 1, incv)
	work.Axpy(n, alpha, v, incv, 1)

	//     C := C - v * w' - w * v'
	if err = c.Syr2(uplo, n, -tau, v, incv, work, 1); err != nil {
		panic(err)
	}
}
