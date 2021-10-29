package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Zptts2 solves a tridiagonal system of the form
//    A * X = B
// using the factorization A = U**H *D*U or A = L*D*L**H computed by ZPTTRF.
// D is a diagonal matrix specified in the vector D, U (or L) is a unit
// bidiagonal matrix whose superdiagonal (subdiagonal) is specified in
// the vector E, and X and B are N by NRHS matrices.
func Zptts2(iuplo, n, nrhs int, d *mat.Vector, e *mat.CVector, b *mat.CMatrix) {
	var i, j int

	//     Quick return if possible
	if n <= 1 {
		if n == 1 {
			goblas.Zdscal(nrhs, 1./d.Get(0), b.CVector(0, 0))
		}
		return
	}

	if iuplo == 1 {
		//        Solve A * X = B using the factorization A = U**H *D*U,
		//        overwriting each right hand side vector with its solution.
		if nrhs <= 2 {
			j = 1
		label10:
			;

			//           Solve U**H * x = b.
			for i = 2; i <= n; i++ {
				b.Set(i-1, j-1, b.Get(i-1, j-1)-b.Get(i-1-1, j-1)*e.GetConj(i-1-1))
			}

			//           Solve D * U * x = b.
			for i = 1; i <= n; i++ {
				b.Set(i-1, j-1, b.Get(i-1, j-1)/d.GetCmplx(i-1))
			}
			for i = n - 1; i >= 1; i-- {
				b.Set(i-1, j-1, b.Get(i-1, j-1)-b.Get(i, j-1)*e.Get(i-1))
			}
			if j < nrhs {
				j = j + 1
				goto label10
			}
		} else {
			for j = 1; j <= nrhs; j++ {
				//              Solve U**H * x = b.
				for i = 2; i <= n; i++ {
					b.Set(i-1, j-1, b.Get(i-1, j-1)-b.Get(i-1-1, j-1)*e.GetConj(i-1-1))
				}

				//              Solve D * U * x = b.
				b.Set(n-1, j-1, b.Get(n-1, j-1)/d.GetCmplx(n-1))
				for i = n - 1; i >= 1; i-- {
					b.Set(i-1, j-1, b.Get(i-1, j-1)/d.GetCmplx(i-1)-b.Get(i, j-1)*e.Get(i-1))
				}
			}
		}
	} else {
		//        Solve A * X = B using the factorization A = L*D*L**H,
		//        overwriting each right hand side vector with its solution.
		if nrhs <= 2 {
			j = 1
		label80:
			;

			//           Solve L * x = b.
			for i = 2; i <= n; i++ {
				b.Set(i-1, j-1, b.Get(i-1, j-1)-b.Get(i-1-1, j-1)*e.Get(i-1-1))
			}

			//           Solve D * L**H * x = b.
			for i = 1; i <= n; i++ {
				b.Set(i-1, j-1, b.Get(i-1, j-1)/d.GetCmplx(i-1))
			}
			for i = n - 1; i >= 1; i-- {
				b.Set(i-1, j-1, b.Get(i-1, j-1)-b.Get(i, j-1)*e.GetConj(i-1))
			}
			if j < nrhs {
				j = j + 1
				goto label80
			}
		} else {
			for j = 1; j <= nrhs; j++ {
				//              Solve L * x = b.
				for i = 2; i <= n; i++ {
					b.Set(i-1, j-1, b.Get(i-1, j-1)-b.Get(i-1-1, j-1)*e.Get(i-1-1))
				}

				//              Solve D * L**H * x = b.
				b.Set(n-1, j-1, b.Get(n-1, j-1)/d.GetCmplx(n-1))
				for i = n - 1; i >= 1; i-- {
					b.Set(i-1, j-1, b.Get(i-1, j-1)/d.GetCmplx(i-1)-b.Get(i, j-1)*e.GetConj(i-1))
				}
			}
		}
	}
}
