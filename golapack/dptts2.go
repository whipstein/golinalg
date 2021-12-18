package golapack

import (
	"github.com/whipstein/golinalg/mat"
)

// Dptts2 solves a tridiagonal system of the form
//    A * X = B
// using the L*D*L**T factorization of A computed by DPTTRF.  D is a
// diagonal matrix specified in the vector D, L is a unit bidiagonal
// matrix whose subdiagonal is specified in the vector E, and X and B
// are N by NRHS matrices.
func Dptts2(n, nrhs int, d, e *mat.Vector, b *mat.Matrix) {
	var i, j int

	//     Quick return if possible
	if n <= 1 {
		if n == 1 {
			b.OffIdx(0).Vector().Scal(nrhs, 1./d.Get(0), b.Rows)
		}
		return
	}

	//     Solve A * X = B using the factorization A = L*D*L**T,
	//     overwriting each right hand side vector with its solution.
	for j = 1; j <= nrhs; j++ {
		//           Solve L * x = b.
		for i = 2; i <= n; i++ {
			b.Set(i-1, j-1, b.Get(i-1, j-1)-b.Get(i-1-1, j-1)*e.Get(i-1-1))
		}

		//           Solve D * L**T * x = b.
		b.Set(n-1, j-1, b.Get(n-1, j-1)/d.Get(n-1))
		for i = n - 1; i >= 1; i-- {
			b.Set(i-1, j-1, b.Get(i-1, j-1)/d.Get(i-1)-b.Get(i, j-1)*e.Get(i-1))
		}
	}

	return
}
