package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dgttrs solves one of the systems of equations
//    A*X = B  or  A**T*X = B,
// with a tridiagonal matrix A using the LU factorization computed
// by DGTTRF.
func Dgttrs(trans byte, n, nrhs *int, dl, d, du, du2 *mat.Vector, ipiv *[]int, b *mat.Matrix, ldb *int, info *int) {
	var notran bool
	var itrans, j, jb, nb int

	(*info) = 0
	notran = trans == 'N' || trans == 'n'
	if !notran && !(trans == 'T' || trans == 't') && !(trans == 'C' || trans == 'c') {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*ldb) < maxint(*n, 1) {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGTTRS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
		return
	}

	//     Decode TRANS
	if notran {
		itrans = 0
	} else {
		itrans = 1
	}

	//     Determine the number of right-hand sides to solve at a time.
	if (*nrhs) == 1 {
		nb = 1
	} else {
		nb = maxint(1, Ilaenv(func() *int { y := 1; return &y }(), []byte("DGTTRS"), []byte{trans}, n, nrhs, toPtr(-1), toPtr(-1)))
	}

	if nb >= (*nrhs) {
		Dgtts2(&itrans, n, nrhs, dl, d, du, du2, ipiv, b, ldb)
	} else {
		for j = 1; j <= (*nrhs); j += nb {
			jb = minint((*nrhs)-j+1, nb)
			Dgtts2(&itrans, n, &jb, dl, d, du, du2, ipiv, b.Off(0, j-1), ldb)
		}
	}
}
