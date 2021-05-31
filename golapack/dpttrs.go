package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpttrs solves a tridiagonal system of the form
//    A * X = B
// using the L*D*L**T factorization of A computed by DPTTRF.  D is a
// diagonal matrix specified in the vector D, L is a unit bidiagonal
// matrix whose subdiagonal is specified in the vector E, and X and B
// are N by NRHS matrices.
func Dpttrs(n, nrhs *int, d, e *mat.Vector, b *mat.Matrix, ldb, info *int) {
	var j, jb, nb int

	//     Test the input arguments.
	(*info) = 0
	if (*n) < 0 {
		(*info) = -1
	} else if (*nrhs) < 0 {
		(*info) = -2
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -6
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DPTTRS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
		return
	}

	//     Determine the number of right-hand sides to solve at a time.
	if (*nrhs) == 1 {
		nb = 1
	} else {
		nb = maxint(1, Ilaenv(func() *int { y := 1; return &y }(), []byte("DPTTRS"), []byte{' '}, n, nrhs, toPtr(-1), toPtr(-1)))
	}

	if nb >= (*nrhs) {
		Dptts2(n, nrhs, d, e, b, ldb)
	} else {
		for j = 1; j <= (*nrhs); j += nb {
			jb = minint((*nrhs)-j+1, nb)
			Dptts2(n, &jb, d, e, b.Off(0, j-1), ldb)
		}
	}
}
