package golapack

import "github.com/whipstein/golinalg/mat"

// Zlaset initializes a 2-D array A to BETA on the diagonal and
// ALPHA on the offdiagonals.
func Zlaset(uplo mat.MatUplo, m, n int, alpha, beta complex128, a *mat.CMatrix) {
	var i, j int

	if uplo == Upper {
		//        Set the diagonal to BETA and the strictly upper triangular
		//        part of the array to ALPHA.
		for j = 2; j <= n; j++ {
			for i = 1; i <= min(j-1, m); i++ {
				a.Set(i-1, j-1, alpha)
			}
		}
		for i = 1; i <= min(n, m); i++ {
			a.Set(i-1, i-1, beta)
		}

	} else if uplo == Lower {
		//        Set the diagonal to BETA and the strictly lower triangular
		//        part of the array to ALPHA.
		for j = 1; j <= min(m, n); j++ {
			for i = j + 1; i <= m; i++ {
				a.Set(i-1, j-1, alpha)
			}
		}
		for i = 1; i <= min(n, m); i++ {
			a.Set(i-1, i-1, beta)
		}

	} else {
		//        Set the array to BETA on the diagonal and ALPHA on the
		//        offdiagonal.
		for j = 1; j <= n; j++ {
			for i = 1; i <= m; i++ {
				a.Set(i-1, j-1, alpha)
			}
		}
		for i = 1; i <= min(m, n); i++ {
			a.Set(i-1, i-1, beta)
		}
	}
}
