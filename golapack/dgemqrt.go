package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgemqrt overwrites the general real M-by-N matrix C with
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
// generated using the compact WY representation as returned by DGEQRT.
//
// Q is of order M if SIDE = 'L' and of order N  if SIDE = 'R'.
func Dgemqrt(side, trans byte, m, n, k, nb *int, v *mat.Matrix, ldv *int, t *mat.Matrix, ldt *int, c *mat.Matrix, ldc *int, work *mat.Vector, info *int) {
	var left, notran, right, tran bool
	var i, ib, kf, ldwork, q int

	//     .. Test the input arguments ..
	(*info) = 0
	left = side == 'L'
	right = side == 'R'
	tran = trans == 'T'
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
		gltest.Xerbla([]byte("DGEMQRT"), -(*info))
		return
	}

	//     .. Quick return if possible ..
	if (*m) == 0 || (*n) == 0 || (*k) == 0 {
		return
	}

	if left && tran {

		for i = 1; i <= (*k); i += *nb {
			ib = min(*nb, (*k)-i+1)
			Dlarfb('L', 'T', 'F', 'C', toPtr((*m)-i+1), n, &ib, v.Off(i-1, i-1), ldv, t.Off(0, i-1), ldt, c.Off(i-1, 0), ldc, work.Matrix(ldwork, opts), &ldwork)
		}

	} else if right && notran {

		for i = 1; i <= (*k); i += *nb {
			ib = min(*nb, (*k)-i+1)
			Dlarfb('R', 'N', 'F', 'C', m, toPtr((*n)-i+1), &ib, v.Off(i-1, i-1), ldv, t.Off(0, i-1), ldt, c.Off(0, i-1), ldc, work.Matrix(ldwork, opts), &ldwork)
		}

	} else if left && notran {

		kf = (((*k)-1)/(*nb))*(*nb) + 1
		for i = kf; i >= 1; i -= *nb {
			ib = min(*nb, (*k)-i+1)
			Dlarfb('L', 'N', 'F', 'C', toPtr((*m)-i+1), n, &ib, v.Off(i-1, i-1), ldv, t.Off(0, i-1), ldt, c.Off(i-1, 0), ldc, work.Matrix(ldwork, opts), &ldwork)
		}

	} else if right && tran {

		kf = (((*k)-1)/(*nb))*(*nb) + 1
		for i = kf; i >= 1; i -= *nb {
			ib = min(*nb, (*k)-i+1)
			Dlarfb('R', 'T', 'F', 'C', m, toPtr((*n)-i+1), &ib, v.Off(i-1, i-1), ldv, t.Off(0, i-1), ldt, c.Off(0, i-1), ldc, work.Matrix(ldwork, opts), &ldwork)
		}

	}
}
