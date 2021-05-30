package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dorml2 overwrites the general real m by n matrix C with
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
//       Q = H(k) . . . H(2) H(1)
//
// as returned by DGELQF. Q is of order m if SIDE = 'L' and of order n
// if SIDE = 'R'.
func Dorml2(side, trans byte, m, n, k *int, a *mat.Matrix, lda *int, tau *mat.Vector, c *mat.Matrix, ldc *int, work *mat.Vector, info *int) {
	var left, notran bool
	var aii, one float64
	var i, i1, i2, i3, ic, jc, mi, ni, nq int

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
	} else if (*lda) < maxint(1, *k) {
		(*info) = -7
	} else if (*ldc) < maxint(1, *m) {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DORML2"), -(*info))
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
		jc = 1
	} else {
		mi = (*m)
		ic = 1
	}

	for _, i = range genIter(i1, i2, i3) {
		if left {
			//           H(i) is applied to C(i:m,1:n)
			mi = (*m) - i + 1
			ic = i
		} else {
			//           H(i) is applied to C(1:m,i:n)
			ni = (*n) - i + 1
			jc = i
		}

		//        Apply H(i)
		aii = a.Get(i-1, i-1)
		a.Set(i-1, i-1, one)
		Dlarf(side, &mi, &ni, a.Vector(i-1, i-1), lda, tau.GetPtr(i-1), c.Off(ic-1, jc-1), ldc, work)
		a.Set(i-1, i-1, aii)
	}
}
