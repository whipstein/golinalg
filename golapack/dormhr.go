package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dormhr overwrites the general real M-by-N matrix C with
//
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      Q * C          C * Q
// TRANS = 'T':      Q**T * C       C * Q**T
//
// where Q is a real orthogonal matrix of order nq, with nq = m if
// SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of
// IHI-ILO elementary reflectors, as returned by DGEHRD:
//
// Q = H(ilo) H(ilo+1) . . . H(ihi-1).
func Dormhr(side mat.MatSide, trans mat.MatTrans, m, n, ilo, ihi int, a *mat.Matrix, tau *mat.Vector, c *mat.Matrix, work *mat.Vector, lwork int) (err error) {
	var left, lquery bool
	var i1, i2, lwkopt, mi, nb, nh, ni, nq, nw int

	//     Test the input arguments
	nh = ihi - ilo
	left = side == Left
	lquery = (lwork == -1)

	//     NQ is the order of Q and NW is the minimum dimension of WORK
	if left {
		nq = m
		nw = n
	} else {
		nq = n
		nw = m
	}
	if !left && side != Right {
		err = fmt.Errorf("!left && side != Right: side=%s", side)
	} else if trans != NoTrans && trans != Trans {
		err = fmt.Errorf("trans != NoTrans && trans != Trans: trans=%s", trans)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if ilo < 1 || ilo > max(1, nq) {
		err = fmt.Errorf("ilo < 1 || ilo > max(1, nq): ilo=%v, nq=%v", ilo, nq)
	} else if ihi < min(ilo, nq) || ihi > nq {
		err = fmt.Errorf("ihi < min(ilo, nq) || ihi > nq: ihi=%v, ilo=%v, nq=%v", ihi, ilo, nq)
	} else if a.Rows < max(1, nq) {
		err = fmt.Errorf("a.Rows < max(1, nq): a.Rows=%v, nq=%v", a.Rows, nq)
	} else if c.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows < max(1, m): c.Rows=%v, m=%v", c.Rows, m)
	} else if lwork < max(1, nw) && !lquery {
		err = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=%v, nw=%v, lquery=%v", lwork, nw, lquery)
	}

	if err == nil {
		if left {
			nb = Ilaenv(1, "Dormqr", []byte{side.Byte(), trans.Byte()}, nh, n, nh, -1)
		} else {
			nb = Ilaenv(1, "Dormqr", []byte{side.Byte(), trans.Byte()}, m, nh, nh, -1)
		}
		lwkopt = max(1, nw) * nb
		work.Set(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("Dormhr", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 || nh == 0 {
		work.Set(0, 1)
		return
	}

	if left {
		mi = nh
		ni = n
		i1 = ilo + 1
		i2 = 1
	} else {
		mi = m
		ni = nh
		i1 = 1
		i2 = ilo + 1
	}

	if err = Dormqr(side, trans, mi, ni, nh, a.Off(ilo, ilo-1), tau.Off(ilo-1), c.Off(i1-1, i2-1), work, lwork); err != nil {
		panic(err)
	}

	work.Set(0, float64(lwkopt))

	return
}
