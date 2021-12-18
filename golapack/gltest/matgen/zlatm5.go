package matgen

import (
	"math/cmplx"

	"github.com/whipstein/golinalg/mat"
)

// Zlatm5 generates matrices involved in the Generalized Sylvester
// equation:
//
//     A * R - L * B = C
//     D * R - L * E = F
//
// They also satisfy (the diagonalization condition)
//
//  [ I -L ] ( [ A  -C ], [ D -F ] ) [ I  R ] = ( [ A    ], [ D    ] )
//  [    I ] ( [     B ]  [    E ] ) [    I ]   ( [    B ]  [    E ] )
func Zlatm5(prtype, m, n int, a, b, c, d, e, f, r, l *mat.CMatrix, alpha float64, qblcka, qblckb int) {
	var half, imeps, one, reeps, twenty, two, zero complex128
	var i, j, k int
	var err error

	one = (1.0 + 0.0*1i)
	two = (2.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)
	half = (0.5 + 0.0*1i)
	twenty = (2.0e+1 + 0.0*1i)

	if prtype == 1 {
		for i = 1; i <= m; i++ {
			for j = 1; j <= m; j++ {
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

		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				if i == j {
					b.Set(i-1, j-1, one-complex(alpha, 0))
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

		for i = 1; i <= m; i++ {
			for j = 1; j <= n; j++ {
				r.Set(i-1, j-1, (half-cmplx.Sin(complex(float64(i/j), 0)))*twenty)
				l.Set(i-1, j-1, r.Get(i-1, j-1))
			}
		}

	} else if prtype == 2 || prtype == 3 {
		for i = 1; i <= m; i++ {
			for j = 1; j <= m; j++ {
				if i <= j {
					a.Set(i-1, j-1, (half-cmplx.Sin(complex(float64(i), 0)))*two)
					d.Set(i-1, j-1, (half-cmplx.Sin(complex(float64(i*j), 0)))*two)
				} else {
					a.Set(i-1, j-1, zero)
					d.Set(i-1, j-1, zero)
				}
			}
		}

		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				if i <= j {
					b.Set(i-1, j-1, (half-cmplx.Sin(complex(float64(i+j), 0)))*two)
					e.Set(i-1, j-1, (half-cmplx.Sin(complex(float64(j), 0)))*two)
				} else {
					b.Set(i-1, j-1, zero)
					e.Set(i-1, j-1, zero)
				}
			}
		}

		for i = 1; i <= m; i++ {
			for j = 1; j <= n; j++ {
				r.Set(i-1, j-1, (half-cmplx.Sin(complex(float64(i*j), 0)))*twenty)
				l.Set(i-1, j-1, (half-cmplx.Sin(complex(float64(i+j), 0)))*twenty)
			}
		}

		if prtype == 3 {
			if qblcka <= 1 {
				qblcka = 2
			}
			for k = 1; k <= m-1; k += qblcka {
				a.Set(k, k, a.Get(k-1, k-1))
				a.Set(k, k-1, -cmplx.Sin(a.Get(k-1, k)))
			}

			if qblckb <= 1 {
				qblckb = 2
			}
			for k = 1; k <= n-1; k += qblckb {
				b.Set(k, k, b.Get(k-1, k-1))
				b.Set(k, k-1, -cmplx.Sin(b.Get(k-1, k)))
			}
		}

	} else if prtype == 4 {
		for i = 1; i <= m; i++ {
			for j = 1; j <= m; j++ {
				a.Set(i-1, j-1, (half-cmplx.Sin(complex(float64(i*j), 0)))*twenty)
				d.Set(i-1, j-1, (half-cmplx.Sin(complex(float64(i+j), 0)))*two)
			}
		}

		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				b.Set(i-1, j-1, (half-cmplx.Sin(complex(float64(i+j), 0)))*twenty)
				e.Set(i-1, j-1, (half-cmplx.Sin(complex(float64(i*j), 0)))*two)
			}
		}

		for i = 1; i <= m; i++ {
			for j = 1; j <= n; j++ {
				r.Set(i-1, j-1, (half-cmplx.Sin(complex(float64(j/i), 0)))*twenty)
				l.Set(i-1, j-1, (half-cmplx.Sin(complex(float64(i*j), 0)))*two)
			}
		}

	} else if prtype >= 5 {
		reeps = half * two * twenty / complex(alpha, 0)
		imeps = (half - two) / complex(alpha, 0)
		for i = 1; i <= m; i++ {
			for j = 1; j <= n; j++ {
				r.Set(i-1, j-1, (half-cmplx.Sin(complex(float64(i*j), 0)))*complex(alpha, 0)/twenty)
				l.Set(i-1, j-1, (half-cmplx.Sin(complex(float64(i+j), 0)))*complex(alpha, 0)/twenty)
			}
		}

		for i = 1; i <= m; i++ {
			d.Set(i-1, i-1, one)
		}

		for i = 1; i <= m; i++ {
			if i <= 4 {
				a.Set(i-1, i-1, one)
				if i > 2 {
					a.Set(i-1, i-1, one+reeps)
				}
				if (i%2) != 0 && i < m {
					a.Set(i-1, i, imeps)
				} else if i > 1 {
					a.Set(i-1, i-1-1, -imeps)
				}
			} else if i <= 8 {
				if i <= 6 {
					a.Set(i-1, i-1, reeps)
				} else {
					a.Set(i-1, i-1, -reeps)
				}
				if (i%2) != 0 && i < m {
					a.Set(i-1, i, one)
				} else if i > 1 {
					a.Set(i-1, i-1-1, -one)
				}
			} else {
				a.Set(i-1, i-1, one)
				if (i%2) != 0 && i < m {
					a.Set(i-1, i, imeps*2)
				} else if i > 1 {
					a.Set(i-1, i-1-1, -imeps*2)
				}
			}
		}

		for i = 1; i <= n; i++ {
			e.Set(i-1, i-1, one)
			if i <= 4 {
				b.Set(i-1, i-1, -one)
				if i > 2 {
					b.Set(i-1, i-1, one-reeps)
				}
				if (i%2) != 0 && i < n {
					b.Set(i-1, i, imeps)
				} else if i > 1 {
					b.Set(i-1, i-1-1, -imeps)
				}
			} else if i <= 8 {
				if i <= 6 {
					b.Set(i-1, i-1, reeps)
				} else {
					b.Set(i-1, i-1, -reeps)
				}
				if (i%2) != 0 && i < n {
					b.Set(i-1, i, one+imeps)
				} else if i > 1 {
					b.Set(i-1, i-1-1, -one-imeps)
				}
			} else {
				b.Set(i-1, i-1, one-reeps)
				if (i%2) != 0 && i < n {
					b.Set(i-1, i, imeps*2)
				} else if i > 1 {
					b.Set(i-1, i-1-1, -imeps*2)
				}
			}
		}
	}

	//     Compute rhs (C, F)
	if err = c.Gemm(NoTrans, NoTrans, m, n, m, one, a, r, zero); err != nil {
		panic(err)
	}
	if err = c.Gemm(NoTrans, NoTrans, m, n, n, -one, l, b, one); err != nil {
		panic(err)
	}
	if err = f.Gemm(NoTrans, NoTrans, m, n, m, one, d, r, zero); err != nil {
		panic(err)
	}
	if err = f.Gemm(NoTrans, NoTrans, m, n, n, -one, l, e, one); err != nil {
		panic(err)
	}
}
