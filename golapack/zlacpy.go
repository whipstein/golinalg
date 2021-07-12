package golapack

import "github.com/whipstein/golinalg/mat"

// Zlacpy copies all or part of a two-dimensional matrix A to another
// matrix B.
func Zlacpy(uplo byte, m, n *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int) {
	var i, j int

	if uplo == 'U' {
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= min(j, *m); i++ {
				b.Set(i-1, j-1, a.Get(i-1, j-1))
			}
		}

	} else if uplo == 'L' {
		for j = 1; j <= (*n); j++ {
			for i = j; i <= (*m); i++ {
				b.Set(i-1, j-1, a.Get(i-1, j-1))
			}
		}

	} else {
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*m); i++ {
				b.Set(i-1, j-1, a.Get(i-1, j-1))
			}
		}
	}
}
