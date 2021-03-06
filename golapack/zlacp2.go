package golapack

import "github.com/whipstein/golinalg/mat"

// Zlacp2 copies all or part of a real two-dimensional matrix A to a
// complex matrix B.
func Zlacp2(uplo mat.MatUplo, m, n int, a *mat.Matrix, b *mat.CMatrix) {
	var i, j int

	if uplo == Upper {
		for j = 1; j <= n; j++ {
			for i = 1; i <= min(j, m); i++ {
				b.SetRe(i-1, j-1, a.Get(i-1, j-1))
			}
		}

	} else if uplo == Lower {
		for j = 1; j <= n; j++ {
			for i = j; i <= m; i++ {
				b.SetRe(i-1, j-1, a.Get(i-1, j-1))
			}
		}

	} else {
		for j = 1; j <= n; j++ {
			for i = 1; i <= m; i++ {
				b.SetRe(i-1, j-1, a.Get(i-1, j-1))
			}
		}
	}
}
