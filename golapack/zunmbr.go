package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zunmbr If VECT = 'Q', Zunmbr overwrites the general complex M-by-N matrix C
// with
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      Q * C          C * Q
// TRANS = 'C':      Q**H * C       C * Q**H
//
// If VECT = 'P', Zunmbr overwrites the general complex M-by-N matrix C
// with
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      P * C          C * P
// TRANS = 'C':      P**H * C       C * P**H
//
// Here Q and P**H are the unitary matrices determined by ZGEBRD when
// reducing a complex matrix A to bidiagonal form: A = Q * B * P**H. Q
// and P**H are defined as products of elementary reflectors H(i) and
// G(i) respectively.
//
// Let nq = m if SIDE = 'L' and nq = n if SIDE = 'R'. Thus nq is the
// order of the unitary matrix Q or P**H that is applied.
//
// If VECT = 'Q', A is assumed to have been an NQ-by-K matrix:
// if nq >= k, Q = H(1) H(2) . . . H(k);
// if nq < k, Q = H(1) H(2) . . . H(nq-1).
//
// If VECT = 'P', A is assumed to have been a K-by-NQ matrix:
// if k < nq, P = G(1) G(2) . . . G(k);
// if k >= nq, P = G(1) G(2) . . . G(nq-1).
func Zunmbr(vect byte, side mat.MatSide, trans mat.MatTrans, m, n, k int, a *mat.CMatrix, tau *mat.CVector, c *mat.CMatrix, work *mat.CVector, lwork int) (err error) {
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
	if m == 0 || n == 0 {
		nw = 0
	}
	if !applyq && vect != 'P' {
		err = fmt.Errorf("!applyq && vect != 'P': vect='%c'", vect)
	} else if !left && side != Right {
		err = fmt.Errorf("!left && side != Right: side=%s", side)
	} else if !notran && trans != ConjTrans {
		err = fmt.Errorf("!notran && trans != ConjTrans: trans=%s", trans)
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
		if nw > 0 {
			if applyq {
				if left {
					nb = Ilaenv(1, "Zunmqr", []byte{side.Byte(), trans.Byte()}, m-1, n, m-1, -1)
				} else {
					nb = Ilaenv(1, "Zunmqr", []byte{side.Byte(), trans.Byte()}, m, n-1, n-1, -1)
				}
			} else {
				if left {
					nb = Ilaenv(1, "Zunmlq", []byte{side.Byte(), trans.Byte()}, m-1, n, m-1, -1)
				} else {
					nb = Ilaenv(1, "Zunmlq", []byte{side.Byte(), trans.Byte()}, m, n-1, n-1, -1)
				}
			}
			lwkopt = max(1, nw*nb)
		} else {
			lwkopt = 1
		}
		work.SetRe(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("Zunmbr", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 {
		return
	}

	if applyq {
		//        Apply Q
		if nq >= k {
			//           Q was determined by a call to ZGEBRD with nq >= k
			if err = Zunmqr(side, trans, m, n, k, a, tau, c, work, lwork); err != nil {
				panic(err)
			}
		} else if nq > 1 {
			//           Q was determined by a call to ZGEBRD with nq < k
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
			if err = Zunmqr(side, trans, mi, ni, nq-1, a.Off(1, 0), tau, c.Off(i1-1, i2-1), work, lwork); err != nil {
				panic(err)
			}
		}
	} else {
		//        Apply P
		if notran {
			transt = ConjTrans
		} else {
			transt = NoTrans
		}
		if nq > k {
			//           P was determined by a call to ZGEBRD with nq > k
			if err = Zunmlq(side, transt, m, n, k, a, tau, c, work, lwork); err != nil {
				panic(err)
			}
		} else if nq > 1 {
			//           P was determined by a call to ZGEBRD with nq <= k
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
			if err = Zunmlq(side, transt, mi, ni, nq-1, a.Off(0, 1), tau, c.Off(i1-1, i2-1), work, lwork); err != nil {
				panic(err)
			}
		}
	}
	work.SetRe(0, float64(lwkopt))

	return
}
