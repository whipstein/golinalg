package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zunml2 overwrites the general complex m-by-n matrix C with
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
//       Q = H(k)**H . . . H(2)**H H(1)**H
//
// as returned by ZGELQF. Q is of order m if SIDE = 'L' and of order n
// if SIDE = 'R'.
func Zunml2(side, trans byte, m, n, k *int, a *mat.CMatrix, lda *int, tau *mat.CVector, c *mat.CMatrix, ldc *int, work *mat.CVector, info *int) {
	var left, notran bool
	var aii, one, taui complex128
	var i, i1, i2, i3, ic, jc, mi, ni, nq int

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
	} else if (*lda) < max(1, *k) {
		(*info) = -7
	} else if (*ldc) < max(1, *m) {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNML2"), -(*info))
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
		jc = 1
	} else {
		mi = (*m)
		ic = 1
	}

	for _, i = range genIter(i1, i2, i3) {
		if left {
			//           H(i) or H(i)**H is applied to C(i:m,1:n)
			mi = (*m) - i + 1
			ic = i
		} else {
			//           H(i) or H(i)**H is applied to C(1:m,i:n)
			ni = (*n) - i + 1
			jc = i
		}

		//        Apply H(i) or H(i)**H
		if notran {
			taui = tau.GetConj(i - 1)
		} else {
			taui = tau.Get(i - 1)
		}
		if i < nq {
			Zlacgv(toPtr(nq-i), a.CVector(i-1, i), lda)
		}
		aii = a.Get(i-1, i-1)
		a.Set(i-1, i-1, one)
		Zlarf(side, &mi, &ni, a.CVector(i-1, i-1), lda, &taui, c.Off(ic-1, jc-1), ldc, work)
		a.Set(i-1, i-1, aii)
		if i < nq {
			Zlacgv(toPtr(nq-i), a.CVector(i-1, i), lda)
		}
	}
}
