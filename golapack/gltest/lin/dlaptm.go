package lin

import "github.com/whipstein/golinalg/mat"

// Dlaptm multiplies an N by NRHS matrix X by a symmetric tridiagonal
// matrix A and stores the result in a matrix B.  The operation has the
// form
//
//    B := alpha * A * X + beta * B
//
// where alpha may be either 1. or -1. and beta may be 0., 1., or -1.
func Dlaptm(n, nrhs *int, alpha *float64, d, e *mat.Vector, x *mat.Matrix, ldx *int, beta *float64, b *mat.Matrix, ldb *int) {
	var one, zero float64
	var i, j int

	one = 1.0
	zero = 0.0

	if (*n) == 0 {
		return
	}

	//     Multiply B by BETA if BETA.NE.1.
	if (*beta) == zero {
		for j = 1; j <= (*nrhs); j++ {
			for i = 1; i <= (*n); i++ {
				b.Set(i-1, j-1, zero)
			}
		}
	} else if (*beta) == -one {
		for j = 1; j <= (*nrhs); j++ {
			for i = 1; i <= (*n); i++ {
				b.Set(i-1, j-1, -b.Get(i-1, j-1))
			}
		}
	}

	if (*alpha) == one {
		//        Compute B := B + A*X
		for j = 1; j <= (*nrhs); j++ {
			if (*n) == 1 {
				b.Set(0, j-1, b.Get(0, j-1)+d.Get(0)*x.Get(0, j-1))
			} else {
				b.Set(0, j-1, b.Get(0, j-1)+d.Get(0)*x.Get(0, j-1)+e.Get(0)*x.Get(1, j-1))
				b.Set((*n)-1, j-1, b.Get((*n)-1, j-1)+e.Get((*n)-1-1)*x.Get((*n)-1-1, j-1)+d.Get((*n)-1)*x.Get((*n)-1, j-1))
				for i = 2; i <= (*n)-1; i++ {
					b.Set(i-1, j-1, b.Get(i-1, j-1)+e.Get(i-1-1)*x.Get(i-1-1, j-1)+d.Get(i-1)*x.Get(i-1, j-1)+e.Get(i-1)*x.Get(i, j-1))
				}
			}
		}
	} else if (*alpha) == -one {
		//        Compute B := B - A*X
		for j = 1; j <= (*nrhs); j++ {
			if (*n) == 1 {
				b.Set(0, j-1, b.Get(0, j-1)-d.Get(0)*x.Get(0, j-1))
			} else {
				b.Set(0, j-1, b.Get(0, j-1)-d.Get(0)*x.Get(0, j-1)-e.Get(0)*x.Get(1, j-1))
				b.Set((*n)-1, j-1, b.Get((*n)-1, j-1)-e.Get((*n)-1-1)*x.Get((*n)-1-1, j-1)-d.Get((*n)-1)*x.Get((*n)-1, j-1))
				for i = 2; i <= (*n)-1; i++ {
					b.Set(i-1, j-1, b.Get(i-1, j-1)-e.Get(i-1-1)*x.Get(i-1-1, j-1)-d.Get(i-1)*x.Get(i-1, j-1)-e.Get(i-1)*x.Get(i, j-1))
				}
			}
		}
	}
}
