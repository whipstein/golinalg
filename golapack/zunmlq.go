package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zunmlq overwrites the general complex M-by-N matrix C with
//
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      Q * C          C * Q
// TRANS = 'C':      Q**H * C       C * Q**H
//
// where Q is a complex unitary matrix defined as the product of k
// elementary reflectors
//
//       Q = H(k)**H . . . H(2)**H H(1)**H
//
// as returned by ZGELQF. Q is of order M if SIDE = 'L' and of order N
// if SIDE = 'R'.
func Zunmlq(side mat.MatSide, trans mat.MatTrans, m, n, k int, a *mat.CMatrix, tau *mat.CVector, c *mat.CMatrix, work *mat.CVector, lwork int) (err error) {
	var left, lquery, notran bool
	var transt mat.MatTrans
	var i, i1, i2, i3, ib, ic, iwt, jc, ldt, ldwork, lwkopt, mi, nb, nbmax, nbmin, ni, nq, nw, tsize int

	nbmax = 64
	ldt = nbmax + 1
	tsize = ldt * nbmax

	//     Test the input arguments
	left = side == Left
	notran = trans == NoTrans
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
	} else if !notran && trans != ConjTrans {
		err = fmt.Errorf("!notran && trans != ConjTrans: trans=%s", trans)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if k < 0 || k > nq {
		err = fmt.Errorf("k < 0 || k > nq: k=%v, nq=%v", k, nq)
	} else if a.Rows < max(1, k) {
		err = fmt.Errorf("a.Rows < max(1, k): a.Rows=%v, k=%v", a.Rows, k)
	} else if c.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows < max(1, m): c.Rows=%v, m=%v", c.Rows, m)
	} else if lwork < max(1, nw) && !lquery {
		err = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=%v, nw=%v, lquery=%v", lwork, nw, lquery)
	}

	if err == nil {
		//        Compute the workspace requirements
		nb = min(nbmax, Ilaenv(1, "Zunmlq", []byte{side.Byte(), trans.Byte()}, m, n, k, -1))
		lwkopt = max(1, nw)*nb + tsize
		work.SetRe(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("Zunmlq", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 || k == 0 {
		work.Set(0, 1)
		return
	}

	nbmin = 2
	ldwork = nw
	if nb > 1 && nb < k {
		if lwork < nw*nb+tsize {
			nb = (lwork - tsize) / ldwork
			nbmin = max(2, Ilaenv(2, "Zunmlq", []byte{side.Byte(), trans.Byte()}, m, n, k, -1))
		}
	}

	if nb < nbmin || nb >= k {
		//        Use unblocked code
		if err = Zunml2(side, trans, m, n, k, a, tau, c, work); err != nil {
			panic(err)
		}
	} else {
		//        Use blocked code
		iwt = 1 + nw*nb
		if (left && notran) || (!left && !notran) {
			i1 = 1
			i2 = k
			i3 = nb
		} else {
			i1 = ((k-1)/nb)*nb + 1
			i2 = 1
			i3 = -nb
		}

		if left {
			ni = n
			jc = 1
		} else {
			mi = m
			ic = 1
		}

		if notran {
			transt = ConjTrans
		} else {
			transt = NoTrans
		}

		for _, i = range genIter(i1, i2, i3) {
			ib = min(nb, k-i+1)

			//           Form the triangular factor of the block reflector
			//           H = H(i) H(i+1) . . . H(i+ib-1)
			Zlarft('F', 'R', nq-i+1, ib, a.Off(i-1, i-1), tau.Off(i-1), work.Off(iwt-1).CMatrix(ldt, opts))
			if left {
				//              H or H**H is applied to C(i:m,1:n)
				mi = m - i + 1
				ic = i
			} else {
				//              H or H**H is applied to C(1:m,i:n)
				ni = n - i + 1
				jc = i
			}

			//           Apply H or H**H
			Zlarfb(side, transt, 'F', 'R', mi, ni, ib, a.Off(i-1, i-1), work.Off(iwt-1).CMatrix(ldt, opts), c.Off(ic-1, jc-1), work.CMatrix(ldwork, opts))
		}
	}
	work.SetRe(0, float64(lwkopt))

	return
}
