package golapack

import (
	"fmt"

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
func Dgemqrt(side mat.MatSide, trans mat.MatTrans, m, n, k, nb int, v, t, c *mat.Matrix, work *mat.Vector) (err error) {
	var left, notran, right, tran bool
	var i, ib, kf, ldwork, q int

	//     .. Test the input arguments ..
	left = side == Left
	right = side == Right
	tran = trans == Trans
	notran = trans == NoTrans

	if left {
		ldwork = max(1, n)
		q = m
	} else if right {
		ldwork = max(1, m)
		q = n
	}
	if !left && !right {
		err = fmt.Errorf("!left && !right: side=%s", side)
	} else if !tran && !notran {
		err = fmt.Errorf("!tran && !notran: trans=%s", trans)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if k < 0 || k > q {
		err = fmt.Errorf("k < 0 || k > q: k=%v, q=%v", k, q)
	} else if nb < 1 || (nb > k && k > 0) {
		err = fmt.Errorf("nb < 1 || (nb > k && k > 0): k=%v, nb=%v", k, nb)
	} else if v.Rows < max(1, q) {
		err = fmt.Errorf("v.Rows < max(1, q): v.Rows=%v, q=%v", v.Rows, q)
	} else if t.Rows < nb {
		err = fmt.Errorf("t.Rows < nb: t.Rows=%v, nb=%v", t.Rows, nb)
	} else if c.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows < max(1, m): c.Rows=%v, m=%v", c.Rows, m)
	}

	if err != nil {
		gltest.Xerbla2("Dgemqrt", err)
		return
	}

	//     .. Quick return if possible ..
	if m == 0 || n == 0 || k == 0 {
		return
	}

	if left && tran {

		for i = 1; i <= k; i += nb {
			ib = min(nb, k-i+1)
			Dlarfb(Left, Trans, 'F', 'C', m-i+1, n, ib, v.Off(i-1, i-1), t.Off(0, i-1), c.Off(i-1, 0), work.Matrix(ldwork, opts))
		}

	} else if right && notran {

		for i = 1; i <= k; i += nb {
			ib = min(nb, k-i+1)
			Dlarfb(Right, NoTrans, 'F', 'C', m, n-i+1, ib, v.Off(i-1, i-1), t.Off(0, i-1), c.Off(0, i-1), work.Matrix(ldwork, opts))
		}

	} else if left && notran {

		kf = ((k-1)/nb)*nb + 1
		for i = kf; i >= 1; i -= nb {
			ib = min(nb, k-i+1)
			Dlarfb(Left, NoTrans, 'F', 'C', m-i+1, n, ib, v.Off(i-1, i-1), t.Off(0, i-1), c.Off(i-1, 0), work.Matrix(ldwork, opts))
		}

	} else if right && tran {

		kf = ((k-1)/nb)*nb + 1
		for i = kf; i >= 1; i -= nb {
			ib = min(nb, k-i+1)
			Dlarfb(Right, Trans, 'F', 'C', m, n-i+1, ib, v.Off(i-1, i-1), t.Off(0, i-1), c.Off(0, i-1), work.Matrix(ldwork, opts))
		}

	}

	return
}
