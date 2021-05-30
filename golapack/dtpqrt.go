package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dtpqrt computes a blocked QR factorization of a real
// "triangular-pentagonal" matrix C, which is composed of a
// triangular block A and pentagonal block B, using the compact
// WY representation for Q.
func Dtpqrt(m, n, l, nb *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, t *mat.Matrix, ldt *int, work *mat.Vector, info *int) {
	var i, ib, iinfo, lb, mb int

	//     Test the input arguments
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*l) < 0 || ((*l) > minint(*m, *n) && minint(*m, *n) >= 0) {
		(*info) = -3
	} else if (*nb) < 1 || ((*nb) > (*n) && (*n) > 0) {
		(*info) = -4
	} else if (*lda) < maxint(1, *n) {
		(*info) = -6
	} else if (*ldb) < maxint(1, *m) {
		(*info) = -8
	} else if (*ldt) < (*nb) {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DTPQRT"), -(*info))
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		return
	}

	for i = 1; i <= (*n); i += (*nb) {
		//     Compute the QR factorization of the current block
		ib = minint((*n)-i+1, *nb)
		mb = minint((*m)-(*l)+i+ib-1, *m)
		if i >= (*l) {
			lb = 0
		} else {
			lb = mb - (*m) + (*l) - i + 1
		}

		Dtpqrt2(&mb, &ib, &lb, a.Off(i-1, i-1), lda, b.Off(0, i-1), ldb, t.Off(0, i-1), ldt, &iinfo)

		//     Update by applying H**T to B(:,I+IB:N) from the left
		if i+ib <= (*n) {
			Dtprfb('L', 'T', 'F', 'C', &mb, toPtr((*n)-i-ib+1), &ib, &lb, b.Off(0, i-1), ldb, t.Off(0, i-1), ldt, a.Off(i-1, i+ib-1), lda, b.Off(0, i+ib-1), ldb, work.Matrix(ib, opts), &ib)
		}
	}
}
