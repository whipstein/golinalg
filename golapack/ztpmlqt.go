package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Ztpmlqt applies a complex orthogonal matrix Q obtained from a
// "triangular-pentagonal" complex block reflector H to a general
// complex matrix C, which consists of two blocks A and B.
func Ztpmlqt(side, trans byte, m, n, k, l, mb *int, v *mat.CMatrix, ldv *int, t *mat.CMatrix, ldt *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, work *mat.CVector, info *int) {
	var left, notran, right, tran bool
	var i, ib, kf, lb, ldaq, nb int

	//     .. Test the input arguments ..
	(*info) = 0
	left = side == 'L'
	right = side == 'R'
	tran = trans == 'C'
	notran = trans == 'N'

	if left {
		ldaq = maxint(1, *k)
	} else if right {
		ldaq = maxint(1, *m)
	}
	if !left && !right {
		(*info) = -1
	} else if !tran && !notran {
		(*info) = -2
	} else if (*m) < 0 {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*k) < 0 {
		(*info) = -5
	} else if (*l) < 0 || (*l) > (*k) {
		(*info) = -6
	} else if (*mb) < 1 || ((*mb) > (*k) && (*k) > 0) {
		(*info) = -7
	} else if (*ldv) < (*k) {
		(*info) = -9
	} else if (*ldt) < (*mb) {
		(*info) = -11
	} else if (*lda) < ldaq {
		(*info) = -13
	} else if (*ldb) < maxint(1, *m) {
		(*info) = -15
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTPMLQT"), -(*info))
		return
	}

	//     .. Quick return if possible ..
	if (*m) == 0 || (*n) == 0 || (*k) == 0 {
		return
	}

	if left && notran {

		for i = 1; i <= (*k); i += (*mb) {
			ib = minint(*mb, (*k)-i+1)
			nb = minint((*m)-(*l)+i+ib-1, *m)
			if i >= (*l) {
				lb = 0
			} else {
				lb = 0
			}
			Ztprfb('L', 'C', 'F', 'R', &nb, n, &ib, &lb, v.Off(i-1, 0), ldv, t.Off(0, i-1), ldt, a.Off(i-1, 0), lda, b, ldb, work.CMatrix(ib, opts), &ib)
		}

	} else if right && tran {

		for i = 1; i <= (*k); i += (*mb) {
			ib = minint(*mb, (*k)-i+1)
			nb = minint((*n)-(*l)+i+ib-1, *n)
			if i >= (*l) {
				lb = 0
			} else {
				lb = nb - (*n) + (*l) - i + 1
			}
			Ztprfb('R', 'N', 'F', 'R', m, &nb, &ib, &lb, v.Off(i-1, 0), ldv, t.Off(0, i-1), ldt, a.Off(0, i-1), lda, b, ldb, work.CMatrix(*m, opts), m)
		}

	} else if left && tran {

		kf = (((*k)-1)/(*mb))*(*mb) + 1
		for i = kf; i >= 1; i -= (*mb) {
			ib = minint(*mb, (*k)-i+1)
			nb = minint((*m)-(*l)+i+ib-1, *m)
			if i >= (*l) {
				lb = 0
			} else {
				lb = 0
			}
			Ztprfb('L', 'N', 'F', 'R', &nb, n, &ib, &lb, v.Off(i-1, 0), ldv, t.Off(0, i-1), ldt, a.Off(i-1, 0), lda, b, ldb, work.CMatrix(ib, opts), &ib)
		}

	} else if right && notran {

		kf = (((*k)-1)/(*mb))*(*mb) + 1
		for i = kf; i >= 1; i -= (*mb) {
			ib = minint(*mb, (*k)-i+1)
			nb = minint((*n)-(*l)+i+ib-1, *n)
			if i >= (*l) {
				lb = 0
			} else {
				lb = nb - (*n) + (*l) - i + 1
			}
			Ztprfb('R', 'C', 'F', 'R', m, &nb, &ib, &lb, v.Off(i-1, 0), ldv, t.Off(0, i-1), ldt, a.Off(0, i-1), lda, b, ldb, work.CMatrix(*m, opts), m)
		}

	}
}
