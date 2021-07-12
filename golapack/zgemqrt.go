package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgemqrt overwrites the general complex M-by-N matrix C with
//
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      Q C            C Q
// TRANS = 'C':    Q**H C            C Q**H
//
// where Q is a complex orthogonal matrix defined as the product of K
// elementary reflectors:
//
//       Q = H(1) H(2) . . . H(K) = I - V T V**H
//
// generated using the compact WY representation as returned by ZGEQRT.
//
// Q is of order M if SIDE = 'L' and of order N  if SIDE = 'R'.
func Zgemqrt(side, trans byte, m, n, k, nb *int, v *mat.CMatrix, ldv *int, t *mat.CMatrix, ldt *int, c *mat.CMatrix, ldc *int, work *mat.CVector, info *int) {
	var left, notran, right, tran bool
	var i, ib, kf, ldwork, q int

	//     .. Test the input arguments ..
	(*info) = 0
	left = side == 'L'
	right = side == 'R'
	tran = trans == 'C'
	notran = trans == 'N'

	if left {
		ldwork = max(1, *n)
		q = (*m)
	} else if right {
		ldwork = max(1, *m)
		q = (*n)
	}
	if !left && !right {
		(*info) = -1
	} else if !tran && !notran {
		(*info) = -2
	} else if (*m) < 0 {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*k) < 0 || (*k) > q {
		(*info) = -5
	} else if (*nb) < 1 || ((*nb) > (*k) && (*k) > 0) {
		(*info) = -6
	} else if (*ldv) < max(1, q) {
		(*info) = -8
	} else if (*ldt) < (*nb) {
		(*info) = -10
	} else if (*ldc) < max(1, *m) {
		(*info) = -12
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGEMQRT"), -(*info))
		return
	}

	//     .. Quick return if possible ..
	if (*m) == 0 || (*n) == 0 || (*k) == 0 {
		return
	}

	if left && tran {

		for i = 1; i <= (*k); i += (*nb) {
			ib = min(*nb, (*k)-i+1)
			Zlarfb('L', 'C', 'F', 'C', toPtr((*m)-i+1), n, &ib, v.Off(i-1, i-1), ldv, t.Off(0, i-1), ldt, c.Off(i-1, 0), ldc, work.CMatrix(ldwork, opts), &ldwork)
		}

	} else if right && notran {

		for i = 1; i <= (*k); i += (*nb) {
			ib = min(*nb, (*k)-i+1)
			Zlarfb('R', 'N', 'F', 'C', m, toPtr((*n)-i+1), &ib, v.Off(i-1, i-1), ldv, t.Off(0, i-1), ldt, c.Off(0, i-1), ldc, work.CMatrix(ldwork, opts), &ldwork)
		}

	} else if left && notran {

		kf = (((*k)-1)/(*nb))*(*nb) + 1
		for i = kf; i >= 1; i -= (*nb) {
			ib = min(*nb, (*k)-i+1)
			Zlarfb('L', 'N', 'F', 'C', toPtr((*m)-i+1), n, &ib, v.Off(i-1, i-1), ldv, t.Off(0, i-1), ldt, c.Off(i-1, 0), ldc, work.CMatrix(ldwork, opts), &ldwork)
		}

	} else if right && tran {

		kf = (((*k)-1)/(*nb))*(*nb) + 1
		for i = kf; i >= 1; i -= (*nb) {
			ib = min(*nb, (*k)-i+1)
			Zlarfb('R', 'C', 'F', 'C', m, toPtr((*n)-i+1), &ib, v.Off(i-1, i-1), ldv, t.Off(0, i-1), ldt, c.Off(0, i-1), ldc, work.CMatrix(ldwork, opts), &ldwork)
		}

	}
}
