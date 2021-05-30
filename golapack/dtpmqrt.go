package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dtpmqrt applies a real orthogonal matrix Q obtained from a
// "triangular-pentagonal" real block reflector H to a general
// real matrix C, which consists of two blocks A and B.
func Dtpmqrt(side, trans byte, m, n, k, l, nb *int, v *mat.Matrix, ldv *int, t *mat.Matrix, ldt *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, work *mat.Vector, info *int) {
	var left, notran, right, tran bool
	var i, ib, kf, lb, ldaq, ldvq, mb int

	//     .. Test the input arguments ..
	(*info) = 0
	left = side == 'L'
	right = side == 'R'
	tran = trans == 'T'
	notran = trans == 'N'

	if left {
		ldvq = maxint(1, *m)
		ldaq = maxint(1, *k)
	} else if right {
		ldvq = maxint(1, *n)
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
	} else if (*nb) < 1 || ((*nb) > (*k) && (*k) > 0) {
		(*info) = -7
	} else if (*ldv) < ldvq {
		(*info) = -9
	} else if (*ldt) < (*nb) {
		(*info) = -11
	} else if (*lda) < ldaq {
		(*info) = -13
	} else if (*ldb) < maxint(1, *m) {
		(*info) = -15
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DTPMQRT"), -(*info))
		return
	}

	//     .. Quick return if possible ..
	if (*m) == 0 || (*n) == 0 || (*k) == 0 {
		return
	}

	if left && tran {

		for i = 1; i <= (*k); i += (*nb) {
			ib = minint(*nb, (*k)-i+1)
			mb = minint((*m)-(*l)+i+ib-1, *m)
			if i >= (*l) {
				lb = 0
			} else {
				lb = mb - (*m) + (*l) - i + 1
			}
			Dtprfb('L', 'T', 'F', 'C', &mb, n, &ib, &lb, v.Off(0, i-1), ldv, t.Off(0, i-1), ldt, a.Off(i-1, 0), lda, b, ldb, work.Matrix(ib, opts), &ib)
		}

	} else if right && notran {

		for i = 1; i <= (*k); i += (*nb) {
			ib = minint(*nb, (*k)-i+1)
			mb = minint((*n)-(*l)+i+ib-1, *n)
			if i >= (*l) {
				lb = 0
			} else {
				lb = mb - (*n) + (*l) - i + 1
			}
			Dtprfb('R', 'N', 'F', 'C', m, &mb, &ib, &lb, v.Off(0, i-1), ldv, t.Off(0, i-1), ldt, a.Off(0, i-1), lda, b, ldb, work.Matrix(*m, opts), m)
		}

	} else if left && notran {

		kf = (((*k)-1)/(*nb))*(*nb) + 1
		for i = kf; i >= 1; i -= (*nb) {
			ib = minint(*nb, (*k)-i+1)
			mb = minint((*m)-(*l)+i+ib-1, *m)
			if i >= (*l) {
				lb = 0
			} else {
				lb = mb - (*m) + (*l) - i + 1
			}
			Dtprfb('L', 'N', 'F', 'C', &mb, n, &ib, &lb, v.Off(0, i-1), ldv, t.Off(0, i-1), ldt, a.Off(i-1, 0), lda, b, ldb, work.Matrix(ib, opts), &ib)
		}

	} else if right && tran {

		kf = (((*k)-1)/(*nb))*(*nb) + 1
		for i = kf; i >= 1; i -= (*nb) {
			ib = minint(*nb, (*k)-i+1)
			mb = minint((*n)-(*l)+i+ib-1, *n)
			if i >= (*l) {
				lb = 0
			} else {
				lb = mb - (*n) + (*l) - i + 1
			}
			Dtprfb('R', 'T', 'F', 'C', m, &mb, &ib, &lb, v.Off(0, i-1), ldv, t.Off(0, i-1), ldt, a.Off(0, i-1), lda, b, ldb, work.Matrix(*m, opts), m)
		}

	}
}
