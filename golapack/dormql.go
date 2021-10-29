package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dormql overwrites the general real M-by-N matrix C with
//
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      Q * C          C * Q
// TRANS = 'T':      Q**T * C       C * Q**T
//
// where Q is a real orthogonal matrix defined as the product of k
// elementary reflectors
//
//       Q = H(k) . . . H(2) H(1)
//
// as returned by DGEQLF. Q is of order M if SIDE = 'L' and of order N
// if SIDE = 'R'.
func Dormql(side mat.MatSide, trans mat.MatTrans, m, n, k int, a *mat.Matrix, tau *mat.Vector, c *mat.Matrix, work *mat.Vector, lwork int) (err error) {
	var left, lquery, notran bool
	var i, i1, i2, i3, ib, iwt, ldt, ldwork, lwkopt, mi, nb, nbmax, nbmin, ni, nq, nw, tsize int

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
		nw = max(1, n)
	} else {
		nq = n
		nw = max(1, m)
	}
	if !left && side != Right {
		err = fmt.Errorf("!left && side != Right: side=%s", side)
	} else if !notran && trans != Trans {
		err = fmt.Errorf("!notran && trans != Trans: trans=%s", trans)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if k < 0 || k > nq {
		err = fmt.Errorf("k < 0 || k > nq: k=%v, nq=%v", k, nq)
	} else if a.Rows < max(1, nq) {
		err = fmt.Errorf("a.Rows < max(1, nq): a.Rows=%v, nq=%v", a.Rows, nq)
	} else if c.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows < max(1, m): c.Rows=%v, m=%v", c.Rows, m)
	} else if lwork < nw && !lquery {
		err = fmt.Errorf("lwork < nw && !lquery: lwork=%v, nw=%v, lquery=%v", lwork, nw, lquery)
	}

	if err == nil {
		//        Compute the workspace requirements
		if m == 0 || n == 0 {
			lwkopt = 1
		} else {
			nb = min(nbmax, Ilaenv(1, "Dormql", []byte{side.Byte(), trans.Byte()}, m, n, k, -1))
			lwkopt = nw*nb + tsize
		}
		work.Set(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("Dormql", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 {
		return
	}

	nbmin = 2
	ldwork = nw
	if nb > 1 && nb < k {
		if lwork < nw*nb+tsize {
			nb = (lwork - tsize) / ldwork
			nbmin = max(2, Ilaenv(2, "Dormql", []byte{side.Byte(), trans.Byte()}, m, n, k, -1))
		}
	}

	if nb < nbmin || nb >= k {
		//        Use unblocked code
		if err = Dorm2l(side, trans, m, n, k, a, tau, c, work); err != nil {
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
		} else {
			mi = m
		}

		for _, i = range genIter(i1, i2, i3) {
			ib = min(nb, k-i+1)

			//           Form the triangular factor of the block reflector
			//           H = H(i+ib-1) . . . H(i+1) H(i)
			Dlarft('B', 'C', nq-k+i+ib-1, ib, a.Off(0, i-1), tau.Off(i-1), work.MatrixOff(iwt-1, ldt, opts))
			if left {
				//              H or H**T is applied to C(1:m-k+i+ib-1,1:n)
				mi = m - k + i + ib - 1
			} else {
				//              H or H**T is applied to C(1:m,1:n-k+i+ib-1)
				ni = n - k + i + ib - 1
			}

			//           Apply H or H**T
			Dlarfb(side, trans, 'B', 'C', mi, ni, ib, a.Off(0, i-1), work.MatrixOff(iwt-1, ldt, opts), c, work.Matrix(ldwork, opts))
		}
	}
	work.Set(0, float64(lwkopt))

	return
}
