package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgemlqt overwrites the general real M-by-N matrix C with
//
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      Q C            C Q
// TRANS = 'C':   Q**H C            C Q**H
//
// where Q is a complex orthogonal matrix defined as the product of K
// elementary reflectors:
//
//       Q = H(1) H(2) . . . H(K) = I - V T V**H
//
// generated using the compact WY representation as returned by ZGELQT.
//
// Q is of order M if SIDE = 'L' and of order N  if SIDE = 'R'.
func Zgemlqt(side, trans byte, m, n, k, mb *int, v *mat.CMatrix, ldv *int, t *mat.CMatrix, ldt *int, c *mat.CMatrix, ldc *int, work *mat.CVector, info *int) {
	var left, notran, right, tran bool
	var i, ib, kf, ldwork int

	//     .. Test the input arguments ..
	(*info) = 0
	left = side == 'L'
	right = side == 'R'
	tran = trans == 'C'
	notran = trans == 'N'

	if left {
		ldwork = max(1, *n)
	} else if right {
		ldwork = max(1, *m)
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
	} else if (*mb) < 1 || ((*mb) > (*k) && (*k) > 0) {
		(*info) = -6
	} else if (*ldv) < max(1, *k) {
		(*info) = -8
	} else if (*ldt) < (*mb) {
		(*info) = -10
	} else if (*ldc) < max(1, *m) {
		(*info) = -12
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGEMLQT"), -(*info))
		return
	}

	//     .. Quick return if possible ..
	if (*m) == 0 || (*n) == 0 || (*k) == 0 {
		return
	}

	if left && notran {

		for i = 1; i <= (*k); i += (*mb) {
			ib = min(*mb, (*k)-i+1)
			Zlarfb('L', 'C', 'F', 'R', toPtr((*m)-i+1), n, &ib, v.Off(i-1, i-1), ldv, t.Off(0, i-1), ldt, c.Off(i-1, 0), ldc, work.CMatrix(ldwork, opts), &ldwork)
		}

	} else if right && tran {

		for i = 1; i <= (*k); i += (*mb) {
			ib = min(*mb, (*k)-i+1)
			Zlarfb('R', 'N', 'F', 'R', m, toPtr((*n)-i+1), &ib, v.Off(i-1, i-1), ldv, t.Off(0, i-1), ldt, c.Off(0, i-1), ldc, work.CMatrix(ldwork, opts), &ldwork)
		}

	} else if left && tran {

		kf = (((*k)-1)/(*mb))*(*mb) + 1
		for i = kf; i >= 1; i -= (*mb) {
			ib = min(*mb, (*k)-i+1)
			Zlarfb('L', 'N', 'F', 'R', toPtr((*m)-i+1), n, &ib, v.Off(i-1, i-1), ldv, t.Off(0, i-1), ldt, c.Off(i-1, 0), ldc, work.CMatrix(ldwork, opts), &ldwork)
		}

	} else if right && notran {

		kf = (((*k)-1)/(*mb))*(*mb) + 1
		for i = kf; i >= 1; i -= (*mb) {
			ib = min(*mb, (*k)-i+1)
			Zlarfb('R', 'C', 'F', 'R', m, toPtr((*n)-i+1), &ib, v.Off(i-1, i-1), ldv, t.Off(0, i-1), ldt, c.Off(0, i-1), ldc, work.CMatrix(ldwork, opts), &ldwork)
		}

	}
}
