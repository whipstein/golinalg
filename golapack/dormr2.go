package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dormr2 overwrites the general real m by n matrix C with
//
//       Q * C  if SIDE = 'L' and TRANS = 'N', or
//
//       Q**T* C  if SIDE = 'L' and TRANS = 'T', or
//
//       C * Q  if SIDE = 'R' and TRANS = 'N', or
//
//       C * Q**T if SIDE = 'R' and TRANS = 'T',
//
// where Q is a real orthogonal matrix defined as the product of k
// elementary reflectors
//
//       Q = H(1) H(2) . . . H(k)
//
// as returned by DGERQF. Q is of order m if SIDE = 'L' and of order n
// if SIDE = 'R'.
func Dormr2(side mat.MatSide, trans mat.MatTrans, m, n, k int, a *mat.Matrix, tau *mat.Vector, c *mat.Matrix, work *mat.Vector) (err error) {
	var left, notran bool
	var aii, one float64
	var i, i1, i2, i3, mi, ni, nq int

	one = 1.0

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
	} else if !notran && trans != Trans {
		err = fmt.Errorf("!notran && trans != Trans: trans=%s", trans)
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
	}
	if err != nil {
		gltest.Xerbla2("Dormr2", err)
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 || k == 0 {
		return
	}

	if (left && !notran) || (!left && notran) {
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
			//           H(i) is applied to C(1:m-k+i,1:n)
			mi = m - k + i
		} else {
			//           H(i) is applied to C(1:m,1:n-k+i)
			ni = n - k + i
		}

		//        Apply H(i)
		aii = a.Get(i-1, nq-k+i-1)
		a.Set(i-1, nq-k+i-1, one)
		Dlarf(side, mi, ni, a.Vector(i-1, 0), tau.Get(i-1), c, work)
		a.Set(i-1, nq-k+i-1, aii)
	}

	return
}
