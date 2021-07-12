package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zpttrs solves a tridiagonal system of the form
//    A * X = B
// using the factorization A = U**H *D* U or A = L*D*L**H computed by ZPTTRF.
// D is a diagonal matrix specified in the vector D, U (or L) is a unit
// bidiagonal matrix whose superdiagonal (subdiagonal) is specified in
// the vector E, and X and B are N by NRHS matrices.
func Zpttrs(uplo byte, n, nrhs *int, d *mat.Vector, e *mat.CVector, b *mat.CMatrix, ldb, info *int) {
	var upper bool
	var iuplo, j, jb, nb int

	//     Test the input arguments.
	(*info) = 0
	upper = uplo == 'U' || uplo == 'u'
	if !upper && !(uplo == 'L' || uplo == 'l') {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*ldb) < max(1, *n) {
		(*info) = -7
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZPTTRS"), -(*info))
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
		nb = max(1, Ilaenv(func() *int { y := 1; return &y }(), []byte("ZPTTRS"), []byte{uplo}, n, nrhs, toPtr(-1), toPtr(-1)))
	}

	//     Decode UPLO
	if upper {
		iuplo = 1
	} else {
		iuplo = 0
	}

	if nb >= (*nrhs) {
		Zptts2(&iuplo, n, nrhs, d, e, b, ldb)
	} else {
		for j = 1; j <= (*nrhs); j += nb {
			jb = min((*nrhs)-j+1, nb)
			Zptts2(&iuplo, n, &jb, d, e, b.Off(0, j-1), ldb)
		}
	}
}
