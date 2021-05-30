package golapack

import "golinalg/mat"

// Zlacp2 copies all or part of a real two-dimensional matrix A to a
// complex matrix B.
func Zlacp2(uplo byte, m, n *int, a *mat.Matrix, lda *int, b *mat.CMatrix, ldb *int) {
	var i, j int

	if uplo == 'U' {
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= minint(j, *m); i++ {
				b.SetRe(i-1, j-1, a.Get(i-1, j-1))
			}
		}

	} else if uplo == 'L' {
		for j = 1; j <= (*n); j++ {
			for i = j; i <= (*m); i++ {
				b.SetRe(i-1, j-1, a.Get(i-1, j-1))
			}
		}

	} else {
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*m); i++ {
				b.SetRe(i-1, j-1, a.Get(i-1, j-1))
			}
		}
	}
}
