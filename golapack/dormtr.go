package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dormtr overwrites the general real M-by-N matrix C with
//
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      Q * C          C * Q
// TRANS = 'T':      Q**T * C       C * Q**T
//
// where Q is a real orthogonal matrix of order nq, with nq = m if
// SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of
// nq-1 elementary reflectors, as returned by DSYTRD:
//
// if UPLO = 'U', Q = H(nq-1) . . . H(2) H(1);
//
// if UPLO = 'L', Q = H(1) H(2) . . . H(nq-1).
func Dormtr(side mat.MatSide, uplo mat.MatUplo, trans mat.MatTrans, m, n int, a *mat.Matrix, tau *mat.Vector, c *mat.Matrix, work *mat.Vector, lwork int) (err error) {
	var left, lquery, upper bool
	var i1, i2, lwkopt, mi, nb, ni, nq, nw int

	//     Test the input arguments
	left = side == Left
	upper = uplo == Upper
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
	} else if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if trans != NoTrans && trans != Trans {
		err = fmt.Errorf("trans != NoTrans && trans != Trans: trans=%s", trans)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, nq) {
		err = fmt.Errorf("a.Rows < max(1, nq): a.Rows=%v, nq=%v", a.Rows, nq)
	} else if c.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows < max(1, m): c.Rows=%v, m=%v", c.Rows, m)
	} else if lwork < max(1, nw) && !lquery {
		err = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=%v, nw=%v, lquery=%v", lwork, nw, lquery)
	}

	if err == nil {
		if upper {
			if left {
				nb = Ilaenv(1, "Dormql", []byte{side.Byte(), trans.Byte()}, m-1, n, m-1, -1)
			} else {
				nb = Ilaenv(1, "Dormql", []byte{side.Byte(), trans.Byte()}, m, n-1, n-1, -1)
			}
		} else {
			if left {
				nb = Ilaenv(1, "Dormqr", []byte{side.Byte(), trans.Byte()}, m-1, n, m-1, -1)
			} else {
				nb = Ilaenv(1, "Dormqr", []byte{side.Byte(), trans.Byte()}, m, n-1, n-1, -1)
			}
		}
		lwkopt = max(1, nw) * nb
		work.Set(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("Dormtr", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 || nq == 1 {
		work.Set(0, 1)
		return
	}

	if left {
		mi = m - 1
		ni = n
	} else {
		mi = m
		ni = n - 1
	}

	if upper {
		//        Q was determined by a call to DSYTRD with UPLO = 'U'
		if err = Dormql(side, trans, mi, ni, nq-1, a.Off(0, 1), tau, c, work, lwork); err != nil {
			panic(err)
		}
	} else {
		//        Q was determined by a call to DSYTRD with UPLO = 'L'
		if left {
			i1 = 2
			i2 = 1
		} else {
			i1 = 1
			i2 = 2
		}
		if err = Dormqr(side, trans, mi, ni, nq-1, a.Off(1, 0), tau, c.Off(i1-1, i2-1), work, lwork); err != nil {
			panic(err)
		}
	}
	work.Set(0, float64(lwkopt))

	return
}
