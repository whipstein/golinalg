package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dorm2l overwrites the general real m by n matrix C with
//
//       Q * C  if SIDE = 'L' and TRANS = 'N', or
//
//       Q**T * C  if SIDE = 'L' and TRANS = 'T', or
//
//       C * Q  if SIDE = 'R' and TRANS = 'N', or
//
//       C * Q**T if SIDE = 'R' and TRANS = 'T',
//
// where Q is a real orthogonal matrix defined as the product of k
// elementary reflectors
//
//       Q = H(k) . . . H(2) H(1)
//
// as returned by DGEQLF. Q is of order m if SIDE = 'L' and of order n
// if SIDE = 'R'.
func Dorm2l(side, trans byte, m, n, k *int, a *mat.Matrix, lda *int, tau *mat.Vector, c *mat.Matrix, ldc *int, work *mat.Vector, info *int) {
	var left, notran bool
	var aii, one float64
	var i, i1, i2, i3, mi, ni, nq int

	one = 1.0

	//     Test the input arguments
	(*info) = 0
	left = side == 'L'
	notran = trans == 'N'

	//     NQ is the order of Q
	if left {
		nq = (*m)
	} else {
		nq = (*n)
	}
	if !left && side != 'R' {
		(*info) = -1
	} else if !notran && trans != 'T' {
		(*info) = -2
	} else if (*m) < 0 {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*k) < 0 || (*k) > nq {
		(*info) = -5
	} else if (*lda) < maxint(1, nq) {
		(*info) = -7
	} else if (*ldc) < maxint(1, *m) {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DORM2L"), -(*info))
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 || (*k) == 0 {
		return
	}

	if (left && notran) || (!left && !notran) {
		i1 = 1
		i2 = (*k)
		i3 = 1
	} else {
		i1 = (*k)
		i2 = 1
		i3 = -1
	}

	if left {
		ni = (*n)
	} else {
		mi = (*m)
	}

	for _, i = range genIter(i1, i2, i3) {
		if left {
			//           H(i) is applied to C(1:m-k+i,1:n)
			mi = (*m) - (*k) + i
		} else {
			//           H(i) is applied to C(1:m,1:n-k+i)
			ni = (*n) - (*k) + i
		}

		//        Apply H(i)
		aii = a.Get(nq-(*k)+i-1, i-1)
		a.Set(nq-(*k)+i-1, i-1, one)
		Dlarf(side, &mi, &ni, a.Vector(0, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1), c, ldc, work)
		a.Set(nq-(*k)+i-1, i-1, aii)
	}
}
