package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zunmr3 overwrites the general complex m by n matrix C with
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
//       Q = H(1) H(2) . . . H(k)
//
// as returned by ZTZRZF. Q is of order m if SIDE = 'L' and of order n
// if SIDE = 'R'.
func Zunmr3(side, trans byte, m, n, k, l *int, a *mat.CMatrix, lda *int, tau *mat.CVector, c *mat.CMatrix, ldc *int, work *mat.CVector, info *int) {
	var left, notran bool
	var taui complex128
	var i, i1, i2, i3, ic, ja, jc, mi, ni, nq int

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
	} else if (*l) < 0 || (left && ((*l) > (*m))) || (!left && ((*l) > (*n))) {
		(*info) = -6
	} else if (*lda) < maxint(1, *k) {
		(*info) = -8
	} else if (*ldc) < maxint(1, *m) {
		(*info) = -11
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNMR3"), -(*info))
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 || (*k) == 0 {
		return
	}

	if left && !notran || !left && notran {
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
		ja = (*m) - (*l) + 1
		jc = 1
	} else {
		mi = (*m)
		ja = (*n) - (*l) + 1
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
			taui = tau.Get(i - 1)
		} else {
			taui = tau.GetConj(i - 1)
		}
		Zlarz(side, &mi, &ni, l, a.CVector(i-1, ja-1), lda, &taui, c.Off(ic-1, jc-1), ldc, work)

	}
}
