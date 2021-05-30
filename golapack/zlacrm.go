package golapack

import (
	"golinalg/goblas"
	"golinalg/mat"
)

// Zlacrm performs a very simple matrix-matrix multiplication:
//          C := A * B,
// where A is M by N and complex; B is N by N and real;
// C is M by N and complex.
func Zlacrm(m, n *int, a *mat.CMatrix, lda *int, b *mat.Matrix, ldb *int, c *mat.CMatrix, ldc *int, rwork *mat.Vector) {
	var one, zero float64
	var i, j, l int

	one = 1.0
	zero = 0.0

	//     Quick return if possible.
	if ((*m) == 0) || ((*n) == 0) {
		return
	}

	for j = 1; j <= (*n); j++ {
		for i = 1; i <= (*m); i++ {
			rwork.Set((j-1)*(*m)+i-1, a.GetRe(i-1, j-1))
		}
	}

	l = (*m)*(*n) + 1
	goblas.Dgemm(NoTrans, NoTrans, m, n, n, &one, rwork.Matrix(*m, opts), m, b, ldb, &zero, rwork.MatrixOff(l-1, *m, opts), m)
	for j = 1; j <= (*n); j++ {
		for i = 1; i <= (*m); i++ {
			c.Set(i-1, j-1, rwork.GetCmplx(l+(j-1)*(*m)+i-1-1))
		}
	}

	for j = 1; j <= (*n); j++ {
		for i = 1; i <= (*m); i++ {
			rwork.Set((j-1)*(*m)+i-1, a.GetIm(i-1, j-1))
		}
	}
	goblas.Dgemm(NoTrans, NoTrans, m, n, n, &one, rwork.Matrix(*m, opts), m, b, ldb, &zero, rwork.MatrixOff(l-1, *m, opts), m)
	for j = 1; j <= (*n); j++ {
		for i = 1; i <= (*m); i++ {
			c.Set(i-1, j-1, complex(c.GetRe(i-1, j-1), rwork.Get(l+(j-1)*(*m)+i-1-1)))
		}
	}
}
