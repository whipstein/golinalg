package golapack

import (
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
func Zlarfy(uplo mat.MatUplo, n int, v *mat.CVector, incv int, tau complex128, c *mat.CMatrix, work *mat.CVector) {
	var alpha, half, one, zero complex128
	var err error

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)
	half = (0.5 + 0.0*1i)

	if tau == zero {
		return
	}

	//     Form  w:= C * v
	if err = work.Hemv(uplo, n, one, c, v, incv, zero, 1); err != nil {
		panic(err)
	}

	alpha = -half * tau * v.Dotc(n, work, 1, incv)
	work.Axpy(n, alpha, v, incv, 1)

	//     C := C - v * w' - w * v'
	if err = c.Her2(uplo, n, -tau, v, incv, work, 1); err != nil {
		panic(err)
	}
}
