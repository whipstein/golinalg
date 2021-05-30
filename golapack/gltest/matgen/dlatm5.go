package matgen

import (
	"golinalg/goblas"
	"golinalg/mat"
	"math"
)

// Dlatm5 generates matrices involved in the Generalized Sylvester
// equation:
//
//     A * R - L * B = C
//     D * R - L * E = F
//
// They also satisfy (the diagonalization condition)
//
//  [ I -L ] ( [ A  -C ], [ D -F ] ) [ I  R ] = ( [ A    ], [ D    ] )
//  [    I ] ( [     B ]  [    E ] ) [    I ]   ( [    B ]  [    E ] )
func Dlatm5(prtype, m, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, c *mat.Matrix, ldc *int, d *mat.Matrix, ldd *int, e *mat.Matrix, lde *int, f *mat.Matrix, ldf *int, r *mat.Matrix, ldr *int, l *mat.Matrix, ldl *int, alpha *float64, qblcka, qblckb *int) {
	var half, imeps, one, reeps, twenty, two, zero float64
	var i, j, k int

	one = 1.0
	zero = 0.0
	twenty = 2.0e+1
	half = 0.5
	two = 2.0

	if (*prtype) == 1 {
		for i = 1; i <= (*m); i++ {
			for j = 1; j <= (*m); j++ {
				if i == j {
					a.Set(i-1, j-1, one)
					d.Set(i-1, j-1, one)
				} else if i == j-1 {
					a.Set(i-1, j-1, -one)
					d.Set(i-1, j-1, zero)
				} else {
					a.Set(i-1, j-1, zero)
					d.Set(i-1, j-1, zero)
				}
			}
		}

		for i = 1; i <= (*n); i++ {
			for j = 1; j <= (*n); j++ {
				if i == j {
					b.Set(i-1, j-1, one-(*alpha))
					e.Set(i-1, j-1, one)
				} else if i == j-1 {
					b.Set(i-1, j-1, one)
					e.Set(i-1, j-1, zero)
				} else {
					b.Set(i-1, j-1, zero)
					e.Set(i-1, j-1, zero)
				}
			}
		}

		for i = 1; i <= (*m); i++ {
			for j = 1; j <= (*n); j++ {
				r.Set(i-1, j-1, (half-math.Sin(float64(i/j)))*twenty)
				l.Set(i-1, j-1, r.Get(i-1, j-1))
			}
		}

	} else if (*prtype) == 2 || (*prtype) == 3 {
		for i = 1; i <= (*m); i++ {
			for j = 1; j <= (*m); j++ {
				if i <= j {
					a.Set(i-1, j-1, (half-math.Sin(float64(i)))*two)
					d.Set(i-1, j-1, (half-math.Sin(float64(i*j)))*two)
				} else {
					a.Set(i-1, j-1, zero)
					d.Set(i-1, j-1, zero)
				}
			}
		}

		for i = 1; i <= (*n); i++ {
			for j = 1; j <= (*n); j++ {
				if i <= j {
					b.Set(i-1, j-1, (half-math.Sin(float64(i+j)))*two)
					e.Set(i-1, j-1, (half-math.Sin(float64(j)))*two)
				} else {
					b.Set(i-1, j-1, zero)
					e.Set(i-1, j-1, zero)
				}
			}
		}

		for i = 1; i <= (*m); i++ {
			for j = 1; j <= (*n); j++ {
				r.Set(i-1, j-1, (half-math.Sin(float64(i*j)))*twenty)
				l.Set(i-1, j-1, (half-math.Sin(float64(i+j)))*twenty)
			}
		}

		if (*prtype) == 3 {
			if (*qblcka) <= 1 {
				(*qblcka) = 2
			}
			for k = 1; k <= (*m)-1; k += (*qblcka) {
				a.Set(k+1-1, k+1-1, a.Get(k-1, k-1))
				a.Set(k+1-1, k-1, -math.Sin(a.Get(k-1, k+1-1)))
			}

			if (*qblckb) <= 1 {
				(*qblckb) = 2
			}
			for k = 1; k <= (*n)-1; k += (*qblckb) {
				b.Set(k+1-1, k+1-1, b.Get(k-1, k-1))
				b.Set(k+1-1, k-1, -math.Sin(b.Get(k-1, k+1-1)))
			}
		}

	} else if (*prtype) == 4 {
		for i = 1; i <= (*m); i++ {
			for j = 1; j <= (*m); j++ {
				a.Set(i-1, j-1, (half-math.Sin(float64(i*j)))*twenty)
				d.Set(i-1, j-1, (half-math.Sin(float64(i+j)))*two)
			}
		}

		for i = 1; i <= (*n); i++ {
			for j = 1; j <= (*n); j++ {
				b.Set(i-1, j-1, (half-math.Sin(float64(i+j)))*twenty)
				e.Set(i-1, j-1, (half-math.Sin(float64(i*j)))*two)
			}
		}

		for i = 1; i <= (*m); i++ {
			for j = 1; j <= (*n); j++ {
				r.Set(i-1, j-1, (half-math.Sin(float64(j/i)))*twenty)
				l.Set(i-1, j-1, (half-math.Sin(float64(i*j)))*two)
			}
		}

	} else if (*prtype) >= 5 {
		reeps = half * two * twenty / (*alpha)
		imeps = (half - two) / (*alpha)
		for i = 1; i <= (*m); i++ {
			for j = 1; j <= (*n); j++ {
				r.Set(i-1, j-1, (half-math.Sin(float64(i*j)))*(*alpha)/twenty)
				l.Set(i-1, j-1, (half-math.Sin(float64(i+j)))*(*alpha)/twenty)
			}
		}

		for i = 1; i <= (*m); i++ {
			d.Set(i-1, i-1, one)
		}

		for i = 1; i <= (*m); i++ {
			if i <= 4 {
				a.Set(i-1, i-1, one)
				if i > 2 {
					a.Set(i-1, i-1, one+reeps)
				}
				if (i%2) != 0 && i < (*m) {
					a.Set(i-1, i+1-1, imeps)
				} else if i > 1 {
					a.Set(i-1, i-1-1, -imeps)
				}
			} else if i <= 8 {
				if i <= 6 {
					a.Set(i-1, i-1, reeps)
				} else {
					a.Set(i-1, i-1, -reeps)
				}
				if (i%2) != 0 && i < (*m) {
					a.Set(i-1, i+1-1, one)
				} else if i > 1 {
					a.Set(i-1, i-1-1, -one)
				}
			} else {
				a.Set(i-1, i-1, one)
				if (i%2) != 0 && i < (*m) {
					a.Set(i-1, i+1-1, imeps*2)
				} else if i > 1 {
					a.Set(i-1, i-1-1, -imeps*2)
				}
			}
		}

		for i = 1; i <= (*n); i++ {
			e.Set(i-1, i-1, one)
			if i <= 4 {
				b.Set(i-1, i-1, -one)
				if i > 2 {
					b.Set(i-1, i-1, one-reeps)
				}
				if (i%2) != 0 && i < (*n) {
					b.Set(i-1, i+1-1, imeps)
				} else if i > 1 {
					b.Set(i-1, i-1-1, -imeps)
				}
			} else if i <= 8 {
				if i <= 6 {
					b.Set(i-1, i-1, reeps)
				} else {
					b.Set(i-1, i-1, -reeps)
				}
				if (i%2) != 0 && i < (*n) {
					b.Set(i-1, i+1-1, one+imeps)
				} else if i > 1 {
					b.Set(i-1, i-1-1, -one-imeps)
				}
			} else {
				b.Set(i-1, i-1, one-reeps)
				if (i%2) != 0 && i < (*n) {
					b.Set(i-1, i+1-1, imeps*2)
				} else if i > 1 {
					b.Set(i-1, i-1-1, -imeps*2)
				}
			}
		}
	}

	//     Compute rhs (C, F)
	goblas.Dgemm(NoTrans, NoTrans, m, n, m, &one, a, lda, r, ldr, &zero, c, ldc)
	goblas.Dgemm(NoTrans, NoTrans, m, n, n, toPtrf64(-one), l, ldl, b, ldb, &one, c, ldc)
	goblas.Dgemm(NoTrans, NoTrans, m, n, m, &one, d, ldd, r, ldr, &zero, f, ldf)
	goblas.Dgemm(NoTrans, NoTrans, m, n, n, toPtrf64(-one), l, ldl, e, lde, &one, f, ldf)
}
