package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgemqr overwrites the general real M-by-N matrix C with
//
//                      SIDE = 'L'     SIDE = 'R'
//      TRANS = 'N':      Q * C          C * Q
//      TRANS = 'T':      Q**T * C       C * Q**T
//
// where Q is a real orthogonal matrix defined as the product
// of blocked elementary reflectors computed by tall skinny
// QR factorization (DGEQR)
func Dgemqr(side mat.MatSide, trans mat.MatTrans, m, n, k int, a *mat.Matrix, t *mat.Vector, tsize int, c *mat.Matrix, work *mat.Vector, lwork int) (err error) {
	var left, lquery, notran, right, tran bool
	var lw, mb, mn, nb int

	//     Test the input arguments
	lquery = lwork == -1
	notran = trans == NoTrans
	tran = trans == Trans
	left = side == Left
	right = side == Right

	mb = int(t.Get(1))
	nb = int(t.Get(2))
	if left {
		lw = n * nb
		mn = m
	} else {
		lw = mb * nb
		mn = n
	}

	// if (mb > k) && (mn > k) {
	// 	if (mn-k)%(mb-k) == 0 {
	// 		nblcks = (mn - k) / (mb - k)
	// 	} else {
	// 		nblcks = (mn-k)/(mb-k) + 1
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
		err = fmt.Errorf("k < 0 || k > mn: k=%v, m=%v, n=%v", k, m, n)
	} else if a.Rows < max(1, mn) {
		err = fmt.Errorf("a.Rows < max(1, mn): a.Rows=%v, m=%v, n=%v", a.Rows, m, n)
	} else if tsize < 5 {
		err = fmt.Errorf("tsize < 5: tsize=%v", tsize)
	} else if c.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows < max(1, m): c.Rows=%v, m=%v", c.Rows, m)
	} else if (lwork < max(1, lw)) && (!lquery) {
		err = fmt.Errorf("(lwork < max(1, lw)) && (!lquery): lwork=%v, lw=%v", lwork, lw)
	}

	if err == nil {
		work.Set(0, float64(lw))
	}

	if err != nil {
		gltest.Xerbla2("Dgemqr", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if min(m, n, k) == 0 {
		return
	}

	if (left && m <= k) || (right && n <= k) || (mb <= k) || (mb >= max(m, n, k)) {
		if err = Dgemqrt(side, trans, m, n, k, nb, a, t.Off(5).Matrix(nb, opts), c, work); err != nil {
			panic(err)
		}
	} else {
		if err = Dlamtsqr(side, trans, m, n, k, mb, nb, a, t.Off(5).Matrix(nb, opts), c, work, lwork); err != nil {
			panic(err)
		}
	}

	work.Set(0, float64(lw))

	return
}
