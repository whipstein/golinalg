package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgemlqt overwrites the general real M-by-N matrix C with
//
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      Q C            C Q
// TRANS = 'T':   Q**T C            C Q**T
//
// where Q is a real orthogonal matrix defined as the product of K
// elementary reflectors:
//
//       Q = H(1) H(2) . . . H(K) = I - V T V**T
//
// generated using the compact WY representation as returned by DGELQT.
//
// Q is of order M if SIDE = 'L' and of order N  if SIDE = 'R'.
func Dgemlqt(side, trans byte, m, n, k, mb *int, v *mat.Matrix, ldv *int, t *mat.Matrix, ldt *int, c *mat.Matrix, ldc *int, work *mat.Vector, info *int) {
	var left, notran, right, tran bool
	var i, ib, kf, ldwork int

	//     .. Test the input arguments ..
	(*info) = 0
	left = side == 'L'
	right = side == 'R'
	tran = trans == 'T'
	notran = trans == 'N'

	if left {
		ldwork = maxint(1, *n)
	} else if right {
		ldwork = maxint(1, *m)
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
	} else if (*ldv) < maxint(1, *k) {
		(*info) = -8
	} else if (*ldt) < (*mb) {
		(*info) = -10
	} else if (*ldc) < maxint(1, *m) {
		(*info) = -12
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGEMLQT"), -(*info))
		return
	}

	//     .. Quick return if possible ..
	if (*m) == 0 || (*n) == 0 || (*k) == 0 {
		return
	}

	if left && notran {

		for i = 1; i <= (*k); i += (*mb) {
			ib = minint(*mb, (*k)-i+1)
			Dlarfb('L', 'T', 'F', 'R', toPtr((*m)-i+1), n, &ib, v.Off(i-1, i-1), ldv, t.Off(0, i-1), ldt, c.Off(i-1, 0), ldc, work.Matrix(ldwork, opts), &ldwork)
		}

	} else if right && tran {

		for i = 1; i <= (*k); i += (*mb) {
			ib = minint(*mb, (*k)-i+1)
			Dlarfb('R', 'N', 'F', 'R', m, toPtr((*n)-i+1), &ib, v.Off(i-1, i-1), ldv, t.Off(0, i-1), ldt, c.Off(0, i-1), ldc, work.Matrix(ldwork, opts), &ldwork)
		}

	} else if left && tran {

		kf = (((*k)-1)/(*mb))*(*mb) + 1
		for i = kf; i >= 1; i -= *mb {
			ib = minint(*mb, (*k)-i+1)
			Dlarfb('L', 'N', 'F', 'R', toPtr((*m)-i+1), n, &ib, v.Off(i-1, i-1), ldv, t.Off(0, i-1), ldt, c.Off(i-1, 0), ldc, work.Matrix(ldwork, opts), &ldwork)
		}

	} else if right && notran {

		kf = (((*k)-1)/(*mb))*(*mb) + 1
		for i = kf; i >= 1; i -= *mb {
			ib = minint(*mb, (*k)-i+1)
			Dlarfb('R', 'T', 'F', 'R', m, toPtr((*n)-i+1), &ib, v.Off(i-1, i-1), ldv, t.Off(0, i-1), ldt, c.Off(0, i-1), ldc, work.Matrix(ldwork, opts), &ldwork)
		}

	}
}
