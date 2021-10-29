package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zunm2l overwrites the general complex m-by-n matrix C with
//
//       Q * C  if SIDE = 'L' and TRANS = 'N', or
//
//       Q**H* C  if SIDE = 'L' and TRANS = 'C', or
//
//       C * Q  if SIDE = 'R' and TRANS = 'N', or
//
//       C * Q**H if SIDE = 'R' and TRANS = 'C',
//
// where Q is a complex unitary matrix defined as the product of k
// elementary reflectors
//
//       Q = H(k) . . . H(2) H(1)
//
// as returned by ZGEQLF. Q is of order m if SIDE = 'L' and of order n
// if SIDE = 'R'.
func Zunm2l(side mat.MatSide, trans mat.MatTrans, m, n, k int, a *mat.CMatrix, tau *mat.CVector, c *mat.CMatrix, work *mat.CVector) (err error) {
	var left, notran bool
	var aii, one, taui complex128
	var i, i1, i2, i3, mi, ni, nq int

	one = (1.0 + 0.0*1i)

	//     Test the input arguments
	left = side == Left
	notran = trans == NoTrans

	//     NQ is the order of Q
	if left {
		nq = m
	} else {
		nq = n
	}
	if !left && side != Right {
		err = fmt.Errorf("!left && side != Right: side=%s", side)
	} else if !notran && trans != ConjTrans {
		err = fmt.Errorf("!notran && trans != ConjTrans: trans=%v", trans)
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
	}
	if err != nil {
		gltest.Xerbla2("Zunm2l", err)
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 || k == 0 {
		return
	}

	if left && notran || !left && !notran {
		i1 = 1
		i2 = k
		i3 = 1
	} else {
		i1 = k
		i2 = 1
		i3 = -1
	}

	if left {
		ni = n
	} else {
		mi = m
	}

	for _, i = range genIter(i1, i2, i3) {
		if left {
			//           H(i) or H(i)**H is applied to C(1:m-k+i,1:n)
			mi = m - k + i
		} else {
			//           H(i) or H(i)**H is applied to C(1:m,1:n-k+i)
			ni = n - k + i
		}

		//        Apply H(i) or H(i)**H
		if notran {
			taui = tau.Get(i - 1)
		} else {
			taui = tau.GetConj(i - 1)
		}
		aii = a.Get(nq-k+i-1, i-1)
		a.Set(nq-k+i-1, i-1, one)
		Zlarf(side, mi, ni, a.CVector(0, i-1, 1), taui, c, work)
		a.Set(nq-k+i-1, i-1, aii)
	}

	return
}
