package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
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
func Zunm2l(side, trans byte, m, n, k *int, a *mat.CMatrix, lda *int, tau *mat.CVector, c *mat.CMatrix, ldc *int, work *mat.CVector, info *int) {
	var left, notran bool
	var aii, one, taui complex128
	var i, i1, i2, i3, mi, ni, nq int

	one = (1.0 + 0.0*1i)

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
	} else if !notran && trans != 'C' {
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
		gltest.Xerbla([]byte("ZUNM2L"), -(*info))
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 || (*k) == 0 {
		return
	}

	if left && notran || !left && !notran {
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
			//           H(i) or H(i)**H is applied to C(1:m-k+i,1:n)
			mi = (*m) - (*k) + i
		} else {
			//           H(i) or H(i)**H is applied to C(1:m,1:n-k+i)
			ni = (*n) - (*k) + i
		}

		//        Apply H(i) or H(i)**H
		if notran {
			taui = tau.Get(i - 1)
		} else {
			taui = tau.GetConj(i - 1)
		}
		aii = a.Get(nq-(*k)+i-1, i-1)
		a.Set(nq-(*k)+i-1, i-1, one)
		Zlarf(side, &mi, &ni, a.CVector(0, i-1), func() *int { y := 1; return &y }(), &taui, c, ldc, work)
		a.Set(nq-(*k)+i-1, i-1, aii)
	}
}
