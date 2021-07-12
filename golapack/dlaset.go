package golapack

import "github.com/whipstein/golinalg/mat"

// Dlaset initializes an m-by-n matrix A to BETA on the diagonal and
// ALPHA on the offdiagonals.
func Dlaset(uplo byte, m, n *int, alpha, beta *float64, a *mat.Matrix, lda *int) {
	var i, j int

	if uplo == 'U' {
		//        Set the strictly upper triangular or trapezoidal part of the
		//        array to ALPHA.
		for j = 2; j <= *n; j++ {
			for i = 1; i <= min(j-1, *m); i++ {
				a.Set(i-1, j-1, *alpha)
			}
		}

	} else if uplo == 'L' {
		//        Set the strictly lower triangular or trapezoidal part of the
		//        array to ALPHA.
		for j = 1; j <= min(*m, *n); j++ {
			for i = j + 1; i <= *m; i++ {
				a.Set(i-1, j-1, *alpha)
			}
		}

	} else {
		//        Set the leading m-by-n submatrix to ALPHA.
		for j = 1; j <= *n; j++ {
			for i = 1; i <= *m; i++ {
				a.Set(i-1, j-1, *alpha)
			}
		}
	}

	//     Set the first min(M,N) diagonal elements to BETA.
	for i = 1; i <= min(*m, *n); i++ {
		a.Set(i-1, i-1, *beta)
	}
}
