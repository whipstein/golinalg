package golapack

import "github.com/whipstein/golinalg/mat"

// Zlagtm performs a matrix-vector product of the form
//
//    B := alpha * A * X + beta * B
//
// where A is a tridiagonal matrix of order N, B and X are N by NRHS
// matrices, and alpha and beta are real scalars, each of which may be
// 0., 1., or -1.
func Zlagtm(trans mat.MatTrans, n, nrhs int, alpha float64, dl, d, du *mat.CVector, x *mat.CMatrix, beta float64, b *mat.CMatrix) {
	var one, zero float64
	var i, j int

	one = 1.0
	zero = 0.0

	if n == 0 {
		return
	}

	//     Multiply B by BETA if BETA.NE.1.
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
		if trans == NoTrans {
			//           Compute B := B + A*X
			for j = 1; j <= nrhs; j++ {
				if n == 1 {
					b.Set(0, j-1, b.Get(0, j-1)+d.Get(0)*x.Get(0, j-1))
				} else {
					b.Set(0, j-1, b.Get(0, j-1)+d.Get(0)*x.Get(0, j-1)+du.Get(0)*x.Get(1, j-1))
					b.Set(n-1, j-1, b.Get(n-1, j-1)+dl.Get(n-1-1)*x.Get(n-1-1, j-1)+d.Get(n-1)*x.Get(n-1, j-1))
					for i = 2; i <= n-1; i++ {
						b.Set(i-1, j-1, b.Get(i-1, j-1)+dl.Get(i-1-1)*x.Get(i-1-1, j-1)+d.Get(i-1)*x.Get(i-1, j-1)+du.Get(i-1)*x.Get(i, j-1))
					}
				}
			}
		} else if trans == Trans {
			//           Compute B := B + A**T * X
			for j = 1; j <= nrhs; j++ {
				if n == 1 {
					b.Set(0, j-1, b.Get(0, j-1)+d.Get(0)*x.Get(0, j-1))
				} else {
					b.Set(0, j-1, b.Get(0, j-1)+d.Get(0)*x.Get(0, j-1)+dl.Get(0)*x.Get(1, j-1))
					b.Set(n-1, j-1, b.Get(n-1, j-1)+du.Get(n-1-1)*x.Get(n-1-1, j-1)+d.Get(n-1)*x.Get(n-1, j-1))
					for i = 2; i <= n-1; i++ {
						b.Set(i-1, j-1, b.Get(i-1, j-1)+du.Get(i-1-1)*x.Get(i-1-1, j-1)+d.Get(i-1)*x.Get(i-1, j-1)+dl.Get(i-1)*x.Get(i, j-1))
					}
				}
			}
		} else if trans == ConjTrans {
			//           Compute B := B + A**H * X
			for j = 1; j <= nrhs; j++ {
				if n == 1 {
					b.Set(0, j-1, b.Get(0, j-1)+d.GetConj(0)*x.Get(0, j-1))
				} else {
					b.Set(0, j-1, b.Get(0, j-1)+d.GetConj(0)*x.Get(0, j-1)+dl.GetConj(0)*x.Get(1, j-1))
					b.Set(n-1, j-1, b.Get(n-1, j-1)+du.GetConj(n-1-1)*x.Get(n-1-1, j-1)+d.GetConj(n-1)*x.Get(n-1, j-1))
					for i = 2; i <= n-1; i++ {
						b.Set(i-1, j-1, b.Get(i-1, j-1)+du.GetConj(i-1-1)*x.Get(i-1-1, j-1)+d.GetConj(i-1)*x.Get(i-1, j-1)+dl.GetConj(i-1)*x.Get(i, j-1))
					}
				}
			}
		}
	} else if alpha == -one {
		if trans == NoTrans {
			//           Compute B := B - A*X
			for j = 1; j <= nrhs; j++ {
				if n == 1 {
					b.Set(0, j-1, b.Get(0, j-1)-d.Get(0)*x.Get(0, j-1))
				} else {
					b.Set(0, j-1, b.Get(0, j-1)-d.Get(0)*x.Get(0, j-1)-du.Get(0)*x.Get(1, j-1))
					b.Set(n-1, j-1, b.Get(n-1, j-1)-dl.Get(n-1-1)*x.Get(n-1-1, j-1)-d.Get(n-1)*x.Get(n-1, j-1))
					for i = 2; i <= n-1; i++ {
						b.Set(i-1, j-1, b.Get(i-1, j-1)-dl.Get(i-1-1)*x.Get(i-1-1, j-1)-d.Get(i-1)*x.Get(i-1, j-1)-du.Get(i-1)*x.Get(i, j-1))
					}
				}
			}
		} else if trans == Trans {
			//           Compute B := B - A**T *X
			for j = 1; j <= nrhs; j++ {
				if n == 1 {
					b.Set(0, j-1, b.Get(0, j-1)-d.Get(0)*x.Get(0, j-1))
				} else {
					b.Set(0, j-1, b.Get(0, j-1)-d.Get(0)*x.Get(0, j-1)-dl.Get(0)*x.Get(1, j-1))
					b.Set(n-1, j-1, b.Get(n-1, j-1)-du.Get(n-1-1)*x.Get(n-1-1, j-1)-d.Get(n-1)*x.Get(n-1, j-1))
					for i = 2; i <= n-1; i++ {
						b.Set(i-1, j-1, b.Get(i-1, j-1)-du.Get(i-1-1)*x.Get(i-1-1, j-1)-d.Get(i-1)*x.Get(i-1, j-1)-dl.Get(i-1)*x.Get(i, j-1))
					}
				}
			}
		} else if trans == ConjTrans {
			//           Compute B := B - A**H *X
			for j = 1; j <= nrhs; j++ {
				if n == 1 {
					b.Set(0, j-1, b.Get(0, j-1)-d.GetConj(0)*x.Get(0, j-1))
				} else {
					b.Set(0, j-1, b.Get(0, j-1)-d.GetConj(0)*x.Get(0, j-1)-dl.GetConj(0)*x.Get(1, j-1))
					b.Set(n-1, j-1, b.Get(n-1, j-1)-du.GetConj(n-1-1)*x.Get(n-1-1, j-1)-d.GetConj(n-1)*x.Get(n-1, j-1))
					for i = 2; i <= n-1; i++ {
						b.Set(i-1, j-1, b.Get(i-1, j-1)-du.GetConj(i-1-1)*x.Get(i-1-1, j-1)-d.GetConj(i-1)*x.Get(i-1, j-1)-dl.GetConj(i-1)*x.Get(i, j-1))
					}
				}
			}
		}
	}
}
