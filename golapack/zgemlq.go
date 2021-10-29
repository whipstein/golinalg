package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgemlq overwrites the general real M-by-N matrix C with
//
//                      SIDE = 'L'     SIDE = 'R'
//      TRANS = 'N':      Q * C          C * Q
//      TRANS = 'C':      Q**H * C       C * Q**H
//      where Q is a complex unitary matrix defined as the product
//      of blocked elementary reflectors computed by short wide
//      LQ factorization (ZGELQ)
func Zgemlq(side mat.MatSide, trans mat.MatTrans, m, n, k int, a *mat.CMatrix, t *mat.CVector, tsize int, c *mat.CMatrix, work *mat.CVector, lwork int) (err error) {
	var left, lquery, notran, right, tran bool
	var lw, mb, mn, nb int

	//     Test the input arguments
	lquery = lwork == -1
	notran = trans == NoTrans
	tran = trans == ConjTrans
	left = side == Left
	right = side == Right

	mb = int(t.GetRe(1))
	nb = int(t.GetRe(2))
	if left {
		lw = n * mb
		mn = m
	} else {
		lw = m * mb
		mn = n
	}

	// if (nb > k) && (mn > k) {
	// 	if (mn-k)%(nb-k) == 0 {
	// 		nblcks = (mn - k) / (nb - k)
	// 	} else {
	// 		nblcks = (mn-k)/(nb-k) + 1
	// 	}
	// } else {
	// 	nblcks = 1
	// }

	if !left && !right {
		err = fmt.Errorf("!left && !right: side=%s", side)
	} else if !tran && !notran {
		err = fmt.Errorf("!tran && !notran: trans=%s", trans)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if k < 0 || k > mn {
		err = fmt.Errorf("k < 0 || k > mn: k=%v, mn=%v", k, mn)
	} else if a.Rows < max(1, k) {
		err = fmt.Errorf("a.Rows < max(1, k): a.Rows=%v, k=%v", a.Rows, k)
	} else if tsize < 5 {
		err = fmt.Errorf("tsize < 5: tsize=%v", tsize)
	} else if c.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows < max(1, m): c.Rows=%v, m=%v", c.Rows, m)
	} else if (lwork < max(1, lw)) && (!lquery) {
		err = fmt.Errorf("(lwork < max(1, lw)) && (!lquery): lwork=%v, lw=%v, lquery=%v", lwork, lw, lquery)
	}

	if err == nil {
		work.SetRe(0, float64(lw))
	}

	if err != nil {
		gltest.Xerbla2("Zgemlq", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if min(m, n, k) == 0 {
		return
	}

	if (left && m <= k) || (right && n <= k) || (nb <= k) || (nb >= max(m, n, k)) {
		if err = Zgemlqt(side, trans, m, n, k, mb, a, t.CMatrixOff(5, mb, opts), c, work); err != nil {
			panic(err)
		}
	} else {
		if err = Zlamswlq(side, trans, m, n, k, mb, nb, a, t.CMatrixOff(5, mb, opts), c, work, lwork); err != nil {
			panic(err)
		}
	}

	work.SetRe(0, float64(lw))

	return
}
