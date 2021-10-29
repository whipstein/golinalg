package golapack

import "github.com/whipstein/golinalg/mat"

// Dlacpy copies all or part of a two-dimensional matrix A to another
// matrix B.
func Dlacpy(uplo mat.MatUplo, m, n int, a, b *mat.Matrix) {
	var i, j int

	if uplo == Upper {
		for j = 1; j <= n; j++ {
			for i = 1; i <= min(j, m); i++ {
				b.Set(i-1, j-1, a.Get(i-1, j-1))
			}
		}
	} else if uplo == Lower {
		for j = 1; j <= n; j++ {
			for i = j; i <= m; i++ {
				b.Set(i-1, j-1, a.Get(i-1, j-1))
			}
		}
	} else {
		for j = 1; j <= n; j++ {
			for i = 1; i <= m; i++ {
				b.Set(i-1, j-1, a.Get(i-1, j-1))
			}
		}
	}
}
