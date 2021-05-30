package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Ztplqt computes a blocked LQ factorization of a complex
// "triangular-pentagonal" matrix C, which is composed of a
// triangular block A and pentagonal block B, using the compact
// WY representation for Q.
func Ztplqt(m, n, l, mb *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, t *mat.CMatrix, ldt *int, work *mat.CVector, info *int) {
	var i, ib, iinfo, lb, nb int

	//     Test the input arguments
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*l) < 0 || ((*l) > minint(*m, *n) && minint(*m, *n) >= 0) {
		(*info) = -3
	} else if (*mb) < 1 || ((*mb) > (*m) && (*m) > 0) {
		(*info) = -4
	} else if (*lda) < maxint(1, *m) {
		(*info) = -6
	} else if (*ldb) < maxint(1, *m) {
		(*info) = -8
	} else if (*ldt) < (*mb) {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTPLQT"), -(*info))
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		return
	}

	for i = 1; i <= (*m); i += (*mb) {
		//     Compute the QR factorization of the current block
		ib = minint((*m)-i+1, *mb)
		nb = minint((*n)-(*l)+i+ib-1, *n)
		if i >= (*l) {
			lb = 0
		} else {
			lb = nb - (*n) + (*l) - i + 1
		}

		Ztplqt2(&ib, &nb, &lb, a.Off(i-1, i-1), lda, b.Off(i-1, 0), ldb, t.Off(0, i-1), ldt, &iinfo)

		//     Update by applying H**T to B(I+IB:M,:) from the right
		if i+ib <= (*m) {
			Ztprfb('R', 'N', 'F', 'R', toPtr((*m)-i-ib+1), &nb, &ib, &lb, b.Off(i-1, 0), ldb, t.Off(0, i-1), ldt, a.Off(i+ib-1, i-1), lda, b.Off(i+ib-1, 0), ldb, work.CMatrix((*m)-i-ib+1, opts), toPtr((*m)-i-ib+1))
		}
	}
}