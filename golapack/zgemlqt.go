package golapack

import (
	"fmt"

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
func Zgemlqt(side mat.MatSide, trans mat.MatTrans, m, n, k, mb int, v, t, c *mat.CMatrix, work *mat.CVector) (err error) {
	var left, notran, right, tran bool
	var i, ib, kf, ldwork int

	//     .. Test the input arguments ..
	left = side == Left
	right = side == Right
	tran = trans == ConjTrans
	notran = trans == NoTrans

	if left {
		ldwork = max(1, n)
	} else if right {
		ldwork = max(1, m)
	}
	if !left && !right {
		err = fmt.Errorf("!left && !right: side=%s", side)
	} else if !tran && !notran {
		err = fmt.Errorf("!tran && !notran: trans=%s", trans)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if k < 0 {
		err = fmt.Errorf("k < 0: k=%v", k)
	} else if mb < 1 || (mb > k && k > 0) {
		err = fmt.Errorf("mb < 1 || (mb > k && k > 0): mb=%v, k=%v", mb, k)
	} else if v.Rows < max(1, k) {
		err = fmt.Errorf("v.Rows < max(1, k): v.Rows=%v, k=%v", v.Rows, k)
	} else if t.Rows < mb {
		err = fmt.Errorf("t.Rows < mb: t.Rows=%v, mb=%v", t.Rows, mb)
	} else if c.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows < max(1, m): c.Rows=%v, m=%v", c.Rows, m)
	}

	if err != nil {
		gltest.Xerbla2("Zgemlqt", err)
		return
	}

	//     .. Quick return if possible ..
	if m == 0 || n == 0 || k == 0 {
		return
	}

	if left && notran {

		for i = 1; i <= k; i += mb {
			ib = min(mb, k-i+1)
			Zlarfb(Left, ConjTrans, 'F', 'R', m-i+1, n, ib, v.Off(i-1, i-1), t.Off(0, i-1), c.Off(i-1, 0), work.CMatrix(ldwork, opts))
		}

	} else if right && tran {

		for i = 1; i <= k; i += mb {
			ib = min(mb, k-i+1)
			Zlarfb(Right, NoTrans, 'F', 'R', m, n-i+1, ib, v.Off(i-1, i-1), t.Off(0, i-1), c.Off(0, i-1), work.CMatrix(ldwork, opts))
		}

	} else if left && tran {

		kf = ((k-1)/mb)*mb + 1
		for i = kf; i >= 1; i -= mb {
			ib = min(mb, k-i+1)
			Zlarfb(Left, NoTrans, 'F', 'R', m-i+1, n, ib, v.Off(i-1, i-1), t.Off(0, i-1), c.Off(i-1, 0), work.CMatrix(ldwork, opts))
		}

	} else if right && notran {

		kf = ((k-1)/mb)*mb + 1
		for i = kf; i >= 1; i -= mb {
			ib = min(mb, k-i+1)
			Zlarfb(Right, ConjTrans, 'F', 'R', m, n-i+1, ib, v.Off(i-1, i-1), t.Off(0, i-1), c.Off(0, i-1), work.CMatrix(ldwork, opts))
		}

	}

	return
}
