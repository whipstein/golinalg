package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgttrs solves one of the systems of equations
//    A * X = B,  A**T * X = B,  or  A**H * X = B,
// with a tridiagonal matrix A using the LU factorization computed
// by ZGTTRF.
func Zgttrs(trans byte, n, nrhs *int, dl, d, du, du2 *mat.CVector, ipiv *[]int, b *mat.CMatrix, ldb, info *int) {
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
	} else if (*ldb) < max(*n, 1) {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGTTRS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
		return
	}

	//     Decode TRANS
	if notran {
		itrans = 0
	} else if trans == 'T' || trans == 't' {
		itrans = 1
	} else {
		itrans = 2
	}

	//     Determine the number of right-hand sides to solve at a time.
	if (*nrhs) == 1 {
		nb = 1
	} else {
		nb = max(1, Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGTTRS"), []byte{trans}, n, nrhs, toPtr(-1), toPtr(-1)))
	}

	if nb >= (*nrhs) {
		Zgtts2(&itrans, n, nrhs, dl, d, du, du2, ipiv, b, ldb)
	} else {
		for j = 1; j <= (*nrhs); j += nb {
			jb = min((*nrhs)-j+1, nb)
			Zgtts2(&itrans, n, &jb, dl, d, du, du2, ipiv, b.Off(0, j-1), ldb)
		}
	}
}
