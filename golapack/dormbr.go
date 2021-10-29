package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dormbr If VECT = 'Q', Dormbr overwrites the general real M-by-N matrix C
// with
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      Q * C          C * Q
// TRANS = 'T':      Q**T * C       C * Q**T
//
// If VECT = 'P', Dormbr overwrites the general real M-by-N matrix C
// with
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      P * C          C * P
// TRANS = 'T':      P**T * C       C * P**T
//
// Here Q and P**T are the orthogonal matrices determined by DGEBRD when
// reducing a real matrix A to bidiagonal form: A = Q * B * P**T. Q and
// P**T are defined as products of elementary reflectors H(i) and G(i)
// respectively.
//
// Let nq = m if SIDE = 'L' and nq = n if SIDE = 'R'. Thus nq is the
// order of the orthogonal matrix Q or P**T that is applied.
//
// If VECT = 'Q', A is assumed to have been an NQ-by-K matrix:
// if nq >= k, Q = H(1) H(2) . . . H(k);
// if nq < k, Q = H(1) H(2) . . . H(nq-1).
//
// If VECT = 'P', A is assumed to have been a K-by-NQ matrix:
// if k < nq, P = G(1) G(2) . . . G(k);
// if k >= nq, P = G(1) G(2) . . . G(nq-1).
func Dormbr(vect byte, side mat.MatSide, trans mat.MatTrans, m, n, k int, a *mat.Matrix, tau *mat.Vector, c *mat.Matrix, work *mat.Vector, lwork int) (err error) {
	var applyq, left, lquery, notran bool
	var transt mat.MatTrans
	var i1, i2, lwkopt, mi, nb, ni, nq, nw int

	//     Test the input arguments
	applyq = vect == 'Q'
	left = side == Left
	notran = trans == NoTrans
	lquery = (lwork == -1)

	//     NQ is the order of Q or P and NW is the minimum dimension of WORK
	if left {
		nq = m
		nw = n
	} else {
		nq = n
		nw = m
	}
	if !applyq && vect != 'P' {
		err = fmt.Errorf("!applyq && vect != 'P': vect='%c'", vect)
	} else if !left && side != Right {
		err = fmt.Errorf("!left && side != Right: side=%s", side)
	} else if !notran && trans != Trans {
		err = fmt.Errorf("!notran && trans != Trans: trans=%s", trans)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if k < 0 {
		err = fmt.Errorf("k < 0: k=%v", k)
	} else if (applyq && a.Rows < max(1, nq)) || (!applyq && a.Rows < max(1, min(nq, k))) {
		err = fmt.Errorf("(applyq && a.Rows < max(1, nq)) || (!applyq && a.Rows < max(1, min(nq, k))): vect='%c', a.Rows=%v, nq=%v, k=%v", vect, a.Rows, nq, k)
	} else if c.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows < max(1, m): c.Rows=%v, m=%v", c.Rows, m)
	} else if lwork < max(1, nw) && !lquery {
		err = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=%v, nw=%v, lquery=%v", lwork, nw, lquery)
	}

	if err == nil {
		if applyq {
			if left {
				nb = Ilaenv(1, "Dormqr", []byte{side.Byte(), trans.Byte()}, m-1, n, m-1, -1)
			} else {
				nb = Ilaenv(1, "Dormqr", []byte{side.Byte(), trans.Byte()}, m, n-1, n-1, -1)
			}
		} else {
			if left {
				nb = Ilaenv(1, "Dormlq", []byte{side.Byte(), trans.Byte()}, m-1, n, m-1, -1)
			} else {
				nb = Ilaenv(1, "Dormlq", []byte{side.Byte(), trans.Byte()}, m, n-1, n-1, -1)
			}
		}
		lwkopt = max(1, nw) * nb
		work.Set(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("Dormbr", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	work.Set(0, 1)
	if m == 0 || n == 0 {
		return
	}

	if applyq {
		//        Apply Q
		if nq >= k {
			//           Q was determined by a call to DGEBRD with nq >= k
			if err = Dormqr(side, trans, m, n, k, a, tau, c, work, lwork); err != nil {
				panic(err)
			}
		} else if nq > 1 {
			//           Q was determined by a call to DGEBRD with nq < k
			if left {
				mi = m - 1
				ni = n
				i1 = 2
				i2 = 1
			} else {
				mi = m
				ni = n - 1
				i1 = 1
				i2 = 2
			}
			if err = Dormqr(side, trans, mi, ni, nq-1, a.Off(1, 0), tau, c.Off(i1-1, i2-1), work, lwork); err != nil {
				panic(err)
			}
		}
	} else {
		//        Apply P
		if notran {
			transt = Trans
		} else {
			transt = NoTrans
		}
		if nq > k {
			//           P was determined by a call to DGEBRD with nq > k
			if err = Dormlq(side, transt, m, n, k, a, tau, c, work, lwork); err != nil {
				panic(err)
			}
		} else if nq > 1 {
			//           P was determined by a call to DGEBRD with nq <= k
			if left {
				mi = m - 1
				ni = n
				i1 = 2
				i2 = 1
			} else {
				mi = m
				ni = n - 1
				i1 = 1
				i2 = 2
			}
			if err = Dormlq(side, transt, mi, ni, nq-1, a.Off(0, 1), tau, c.Off(i1-1, i2-1), work, lwork); err != nil {
				panic(err)
			}
		}
	}
	work.Set(0, float64(lwkopt))

	return
}
