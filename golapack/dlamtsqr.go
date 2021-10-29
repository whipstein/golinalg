package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlamtsqr overwrites the general real M-by-N matrix C with
//
//
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      Q * C          C * Q
// TRANS = 'T':      Q**T * C       C * Q**T
//      where Q is a real orthogonal matrix defined as the product
//      of blocked elementary reflectors computed by tall skinny
//      QR factorization (DLATSQR)
func Dlamtsqr(side mat.MatSide, trans mat.MatTrans, m, n, k, mb, nb int, a, t, c *mat.Matrix, work *mat.Vector, lwork int) (err error) {
	var left, lquery, notran, right, tran bool
	var ctr, i, ii, kk, lw int

	//     Test the input arguments
	lquery = lwork < 0
	notran = trans == NoTrans
	tran = trans == Trans
	left = side == Left
	right = side == Right
	if left {
		lw = n * nb
	} else {
		lw = mb * nb
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
	} else if t.Rows < max(1, nb) {
		err = fmt.Errorf("t.Rows < max(1, nb): t.Rows=%v, nb=%v", t.Rows, nb)
	} else if c.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows < max(1, m): c.Rows=%v, m=%v", c.Rows, m)
	} else if (lwork < max(1, lw)) && (!lquery) {
		err = fmt.Errorf("(lwork < max(1, lw)) && (!lquery): lwork=%v, lw=%v, lquery=%v", lwork, lw, lquery)
	}

	//     Determine the block size if it is tall skinny or short and wide
	if err == nil {
		work.Set(0, float64(lw))
	}

	if err != nil {
		gltest.Xerbla2("Dlamtsqr", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if min(m, n, k) == 0 {
		return
	}

	if (mb <= k) || (mb >= max(m, n, k)) {
		if err = Dgemqrt(side, trans, m, n, k, nb, a, t, c, work); err != nil {
			panic(err)
		}
		return
	}

	if left && notran {
		//         Multiply Q to the last block of C
		kk = (m - k) % (mb - k)
		ctr = (m - k) / (mb - k)
		if kk > 0 {
			ii = m - kk + 1
			if err = Dtpmqrt(Left, NoTrans, kk, n, k, 0, nb, a.Off(ii-1, 0), t.Off(0, ctr*k), c.Off(0, 0), c.Off(ii-1, 0), work); err != nil {
				panic(err)
			}
		} else {
			ii = m + 1
		}

		for i = ii - (mb - k); i >= mb+1; i -= (mb - k) {
			//         Multiply Q to the current block of C (I:I+MB,1:N)
			ctr = ctr - 1
			if err = Dtpmqrt(Left, NoTrans, mb-k, n, k, 0, nb, a.Off(i-1, 0), t.Off(0, ctr*k), c.Off(0, 0), c.Off(i-1, 0), work); err != nil {
				panic(err)
			}

		}

		//         Multiply Q to the first block of C (1:MB,1:N)
		if err = Dgemqrt(Left, NoTrans, mb, n, k, nb, a, t, c, work); err != nil {
			panic(err)
		}

	} else if left && tran {
		//         Multiply Q to the first block of C
		kk = (m - k) % (mb - k)
		ii = m - kk + 1
		ctr = 1
		if err = Dgemqrt(Left, Trans, mb, n, k, nb, a, t, c, work); err != nil {
			panic(err)
		}

		for i = mb + 1; i <= ii-mb+k; i += (mb - k) {
			//         Multiply Q to the current block of C (I:I+MB,1:N)
			if err = Dtpmqrt(Left, Trans, mb-k, n, k, 0, nb, a.Off(i-1, 0), t.Off(0, ctr*k), c.Off(0, 0), c.Off(i-1, 0), work); err != nil {
				panic(err)
			}
			ctr = ctr + 1

		}
		if ii <= m {
			//         Multiply Q to the last block of C
			if err = Dtpmqrt(Left, Trans, kk, n, k, 0, nb, a.Off(ii-1, 0), t.Off(0, ctr*k), c.Off(0, 0), c.Off(ii-1, 0), work); err != nil {
				panic(err)
			}

		}

	} else if right && tran {
		//         Multiply Q to the last block of C
		kk = (n - k) % (mb - k)
		ctr = (n - k) / (mb - k)
		if kk > 0 {
			ii = n - kk + 1
			if err = Dtpmqrt(Right, Trans, m, kk, k, 0, nb, a.Off(ii-1, 0), t.Off(0, ctr*k), c.Off(0, 0), c.Off(0, ii-1), work); err != nil {
				panic(err)
			}
		} else {
			ii = n + 1
		}

		for i = ii - (mb - k); i >= mb+1; i -= (mb - k) {
			//         Multiply Q to the current block of C (1:M,I:I+MB)
			ctr = ctr - 1
			if err = Dtpmqrt(Right, Trans, m, mb-k, k, 0, nb, a.Off(i-1, 0), t.Off(0, ctr*k), c.Off(0, 0), c.Off(0, i-1), work); err != nil {
				panic(err)
			}

		}

		//         Multiply Q to the first block of C (1:M,1:MB)
		if err = Dgemqrt(Right, Trans, m, mb, k, nb, a, t, c, work); err != nil {
			panic(err)
		}

	} else if right && notran {
		//         Multiply Q to the first block of C
		kk = (n - k) % (mb - k)
		ii = n - kk + 1
		ctr = 1
		if err = Dgemqrt(Right, NoTrans, m, mb, k, nb, a, t, c, work); err != nil {
			panic(err)
		}

		for i = mb + 1; i <= ii-mb+k; i += (mb - k) {
			//         Multiply Q to the current block of C (1:M,I:I+MB)
			if err = Dtpmqrt(Right, NoTrans, m, mb-k, k, 0, nb, a.Off(i-1, 0), t.Off(0, ctr*k), c.Off(0, 0), c.Off(0, i-1), work); err != nil {
				panic(err)
			}
			ctr = ctr + 1

		}
		if ii <= n {
			//         Multiply Q to the last block of C
			if err = Dtpmqrt(Right, NoTrans, m, kk, k, 0, nb, a.Off(ii-1, 0), t.Off(0, ctr*k), c.Off(0, 0), c.Off(0, ii-1), work); err != nil {
				panic(err)
			}

		}

	}

	work.Set(0, float64(lw))

	return
}
