package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlamswlq overwrites the general real M-by-N matrix C with
//
//
//                    SIDE = 'L'     SIDE = 'R'
//    TRANS = 'N':      Q * C          C * Q
//    TRANS = 'C':      Q**H * C       C * Q**H
//    where Q is a real orthogonal matrix defined as the product of blocked
//    elementary reflectors computed by short wide LQ
//    factorization (ZLASWLQ)
func Zlamswlq(side mat.MatSide, trans mat.MatTrans, m, n, k, mb, nb int, a, t, c *mat.CMatrix, work *mat.CVector, lwork int) (err error) {
	var left, lquery, notran, right, tran bool
	var ctr, i, ii, kk, lw int

	//     Test the input arguments
	lquery = lwork < 0
	notran = trans == NoTrans
	tran = trans == ConjTrans
	left = side == Left
	right = side == Right
	if left {
		lw = n * mb
	} else {
		lw = m * mb
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
	} else if a.Rows < max(1, k) {
		err = fmt.Errorf("a.Rows < max(1, k): a.Rows=%v, k=%v", a.Rows, k)
	} else if t.Rows < max(1, mb) {
		err = fmt.Errorf("t.Rows < max(1, mb): t.Rows=%v, mb=%v", t.Rows, mb)
	} else if c.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows < max(1, m): c.Rows=%v, m=%v", c.Rows, m)
	} else if (lwork < max(1, lw)) && (!lquery) {
		err = fmt.Errorf("(lwork < max(1, lw)) && (!lquery): lwork=%v, lw=%v, lquery=%v", lwork, lw, lquery)
	}

	if err != nil {
		gltest.Xerbla2("Zlamswlq", err)
		work.SetRe(0, float64(lw))
		return
	} else if lquery {
		work.SetRe(0, float64(lw))
		return
	}

	//     Quick return if possible
	if min(m, n, k) == 0 {
		return
	}

	if (nb <= k) || (nb >= max(m, n, k)) {
		if err = Zgemlqt(side, trans, m, n, k, mb, a, t, c, work); err != nil {
			panic(err)
		}
		return
	}

	if left && tran {
		//         Multiply Q to the last block of C
		kk = (m - k) % (nb - k)
		ctr = (m - k) / (nb - k)

		if kk > 0 {
			ii = m - kk + 1
			if err = Ztpmlqt(Left, ConjTrans, kk, n, k, 0, mb, a.Off(0, ii-1), t.Off(0, ctr*k), c, c.Off(ii-1, 0), work); err != nil {
				panic(err)
			}
		} else {
			ii = m + 1
		}

		for i = ii - (nb - k); i >= nb+1; i -= (nb - k) {
			//         Multiply Q to the current block of C (1:M,I:I+NB)
			ctr = ctr - 1
			if err = Ztpmlqt(Left, ConjTrans, nb-k, n, k, 0, mb, a.Off(0, i-1), t.Off(0, ctr*k), c, c.Off(i-1, 0), work); err != nil {
				panic(err)
			}
		}

		//         Multiply Q to the first block of C (1:M,1:NB)
		if err = Zgemlqt(Left, ConjTrans, nb, n, k, mb, a, t, c, work); err != nil {
			panic(err)
		}

	} else if left && notran {
		//         Multiply Q to the first block of C
		kk = (m - k) % (nb - k)
		ii = m - kk + 1
		ctr = 1
		if err = Zgemlqt(Left, NoTrans, nb, n, k, mb, a, t, c, work); err != nil {
			panic(err)
		}

		for i = nb + 1; i <= ii-nb+k; i += (nb - k) {
			//         Multiply Q to the current block of C (I:I+NB,1:N)
			if err = Ztpmlqt(Left, NoTrans, nb-k, n, k, 0, mb, a.Off(0, i-1), t.Off(0, ctr*k), c, c.Off(i-1, 0), work); err != nil {
				panic(err)
			}
			ctr = ctr + 1

		}
		if ii <= m {
			//         Multiply Q to the last block of C
			if err = Ztpmlqt(Left, NoTrans, kk, n, k, 0, mb, a.Off(0, ii-1), t.Off(0, ctr*k), c, c.Off(ii-1, 0), work); err != nil {
				panic(err)
			}

		}

	} else if right && notran {
		//         Multiply Q to the last block of C
		kk = (n - k) % (nb - k)
		ctr = (n - k) / (nb - k)
		if kk > 0 {
			ii = n - kk + 1
			if err = Ztpmlqt(Right, NoTrans, m, kk, k, 0, mb, a.Off(0, ii-1), t.Off(0, ctr*k), c, c.Off(0, ii-1), work); err != nil {
				panic(err)
			}
		} else {
			ii = n + 1
		}

		for i = ii - (nb - k); i >= nb+1; i -= (nb - k) {
			//         Multiply Q to the current block of C (1:M,I:I+MB)
			ctr = ctr - 1
			if err = Ztpmlqt(Right, NoTrans, m, nb-k, k, 0, mb, a.Off(0, i-1), t.Off(0, ctr*k), c, c.Off(0, i-1), work); err != nil {
				panic(err)
			}
		}

		//         Multiply Q to the first block of C (1:M,1:MB)
		if err = Zgemlqt(Right, NoTrans, m, nb, k, mb, a, t, c, work); err != nil {
			panic(err)
		}

	} else if right && tran {
		//       Multiply Q to the first block of C
		kk = (n - k) % (nb - k)
		ii = n - kk + 1
		if err = Zgemlqt(Right, ConjTrans, m, nb, k, mb, a, t, c, work); err != nil {
			panic(err)
		}
		ctr = 1

		for i = nb + 1; i <= ii-nb+k; i += (nb - k) {
			//         Multiply Q to the current block of C (1:M,I:I+MB)
			if err = Ztpmlqt(Right, ConjTrans, m, nb-k, k, 0, mb, a.Off(0, i-1), t.Off(0, ctr*k), c, c.Off(0, i-1), work); err != nil {
				panic(err)
			}
			ctr = ctr + 1

		}
		if ii <= n {
			//       Multiply Q to the last block of C
			if err = Ztpmlqt(Right, ConjTrans, m, kk, k, 0, mb, a.Off(0, ii-1), t.Off(0, ctr*k), c, c.Off(0, ii-1), work); err != nil {
				panic(err)
			}

		}

	}

	work.SetRe(0, float64(lw))

	return
}
