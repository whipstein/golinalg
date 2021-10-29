package lin

import "github.com/whipstein/golinalg/mat"

// zlaptm multiplies an N by NRHS matrix X by a Hermitian tridiagonal
// matrix A and stores the result in a matrix B.  The operation has the
// form
//
//    B := alpha * A * X + beta * B
//
// where alpha may be either 1. or -1. and beta may be 0., 1., or -1.
func zlaptm(uplo mat.MatUplo, n, nrhs int, alpha float64, d *mat.Vector, e *mat.CVector, x *mat.CMatrix, beta float64, b *mat.CMatrix) {
	var one, zero float64
	var i, j int

	one = 1.0
	zero = 0.0

	if n == 0 {
		return
	}

	if beta == zero {
		for j = 1; j <= nrhs; j++ {
			for i = 1; i <= n; i++ {
				b.SetRe(i-1, j-1, zero)
			}
		}
	} else if beta == -one {
		for j = 1; j <= nrhs; j++ {
			for i = 1; i <= n; i++ {
				b.Set(i-1, j-1, -b.Get(i-1, j-1))
			}
		}
	}

	if alpha == one {
		if uplo == Upper {
			//           Compute B := B + A*X, where E is the superdiagonal of A.
			for j = 1; j <= nrhs; j++ {
				if n == 1 {
					b.Set(0, j-1, b.Get(0, j-1)+d.GetCmplx(0)*x.Get(0, j-1))
				} else {
					b.Set(0, j-1, b.Get(0, j-1)+d.GetCmplx(0)*x.Get(0, j-1)+e.Get(0)*x.Get(1, j-1))
					b.Set(n-1, j-1, b.Get(n-1, j-1)+e.GetConj(n-1-1)*x.Get(n-1-1, j-1)+d.GetCmplx(n-1)*x.Get(n-1, j-1))
					for i = 2; i <= n-1; i++ {
						b.Set(i-1, j-1, b.Get(i-1, j-1)+e.GetConj(i-1-1)*x.Get(i-1-1, j-1)+d.GetCmplx(i-1)*x.Get(i-1, j-1)+e.Get(i-1)*x.Get(i, j-1))
					}
				}
			}
		} else {
			//           Compute B := B + A*X, where E is the subdiagonal of A.
			for j = 1; j <= nrhs; j++ {
				if n == 1 {
					b.Set(0, j-1, b.Get(0, j-1)+d.GetCmplx(0)*x.Get(0, j-1))
				} else {
					b.Set(0, j-1, b.Get(0, j-1)+d.GetCmplx(0)*x.Get(0, j-1)+e.GetConj(0)*x.Get(1, j-1))
					b.Set(n-1, j-1, b.Get(n-1, j-1)+e.Get(n-1-1)*x.Get(n-1-1, j-1)+d.GetCmplx(n-1)*x.Get(n-1, j-1))
					for i = 2; i <= n-1; i++ {
						b.Set(i-1, j-1, b.Get(i-1, j-1)+e.Get(i-1-1)*x.Get(i-1-1, j-1)+d.GetCmplx(i-1)*x.Get(i-1, j-1)+e.GetConj(i-1)*x.Get(i, j-1))
					}
				}
			}
		}
	} else if alpha == -one {
		if uplo == Upper {
			//           Compute B := B - A*X, where E is the superdiagonal of A.
			for j = 1; j <= nrhs; j++ {
				if n == 1 {
					b.Set(0, j-1, b.Get(0, j-1)-d.GetCmplx(0)*x.Get(0, j-1))
				} else {
					b.Set(0, j-1, b.Get(0, j-1)-d.GetCmplx(0)*x.Get(0, j-1)-e.Get(0)*x.Get(1, j-1))
					b.Set(n-1, j-1, b.Get(n-1, j-1)-e.GetConj(n-1-1)*x.Get(n-1-1, j-1)-d.GetCmplx(n-1)*x.Get(n-1, j-1))
					for i = 2; i <= n-1; i++ {
						b.Set(i-1, j-1, b.Get(i-1, j-1)-e.GetConj(i-1-1)*x.Get(i-1-1, j-1)-d.GetCmplx(i-1)*x.Get(i-1, j-1)-e.Get(i-1)*x.Get(i, j-1))
					}
				}
			}
		} else {
			//           Compute B := B - A*X, where E is the subdiagonal of A.
			for j = 1; j <= nrhs; j++ {
				if n == 1 {
					b.Set(0, j-1, b.Get(0, j-1)-d.GetCmplx(0)*x.Get(0, j-1))
				} else {
					b.Set(0, j-1, b.Get(0, j-1)-d.GetCmplx(0)*x.Get(0, j-1)-e.GetConj(0)*x.Get(1, j-1))
					b.Set(n-1, j-1, b.Get(n-1, j-1)-e.Get(n-1-1)*x.Get(n-1-1, j-1)-d.GetCmplx(n-1)*x.Get(n-1, j-1))
					for i = 2; i <= n-1; i++ {
						b.Set(i-1, j-1, b.Get(i-1, j-1)-e.Get(i-1-1)*x.Get(i-1-1, j-1)-d.GetCmplx(i-1)*x.Get(i-1, j-1)-e.GetConj(i-1)*x.Get(i, j-1))
					}
				}
			}
		}
	}
}
