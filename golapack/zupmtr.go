package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zupmtr overwrites the general complex M-by-N matrix C with
//
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      Q * C          C * Q
// TRANS = 'C':      Q**H * C       C * Q**H
//
// where Q is a complex unitary matrix of order nq, with nq = m if
// SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of
// nq-1 elementary reflectors, as returned by ZHPTRD using packed
// storage:
//
// if UPLO = 'U', Q = H(nq-1) . . . H(2) H(1);
//
// if UPLO = 'L', Q = H(1) H(2) . . . H(nq-1).
func Zupmtr(side, uplo, trans byte, m, n *int, ap, tau *mat.CVector, c *mat.CMatrix, ldc *int, work *mat.CVector, info *int) {
	var forwrd, left, notran, upper bool
	var aii, one, taui complex128
	var i, i1, i2, i3, ic, ii, jc, mi, ni, nq int

	one = (1.0 + 0.0*1i)

	//     Test the input arguments
	(*info) = 0
	left = side == 'L'
	notran = trans == 'N'
	upper = uplo == 'U'

	//     NQ is the order of Q
	if left {
		nq = (*m)
	} else {
		nq = (*n)
	}
	if !left && side != 'R' {
		(*info) = -1
	} else if !upper && uplo != 'L' {
		(*info) = -2
	} else if !notran && trans != 'C' {
		(*info) = -3
	} else if (*m) < 0 {
		(*info) = -4
	} else if (*n) < 0 {
		(*info) = -5
	} else if (*ldc) < max(1, *m) {
		(*info) = -9
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUPMTR"), -(*info))
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		return
	}

	if upper {
		//        Q was determined by a call to ZHPTRD with UPLO = 'U'
		forwrd = (left && notran) || (!left && !notran)

		if forwrd {
			i1 = 1
			i2 = nq - 1
			i3 = 1
			ii = 2
		} else {
			i1 = nq - 1
			i2 = 1
			i3 = -1
			ii = nq*(nq+1)/2 - 1
		}

		if left {
			ni = (*n)
		} else {
			mi = (*m)
		}
		//
		for _, i = range genIter(i1, i2, i3) {
			if left {
				//              H(i) or H(i)**H is applied to C(1:i,1:n)
				mi = i
			} else {
				//              H(i) or H(i)**H is applied to C(1:m,1:i)
				ni = i
			}

			//           Apply H(i) or H(i)**H
			if notran {
				taui = tau.Get(i - 1)
			} else {
				taui = tau.GetConj(i - 1)
			}
			aii = ap.Get(ii - 1)
			ap.Set(ii-1, one)
			Zlarf(side, &mi, &ni, ap.Off(ii-i), func() *int { y := 1; return &y }(), &taui, c, ldc, work)
			ap.Set(ii-1, aii)

			if forwrd {
				ii = ii + i + 2
			} else {
				ii = ii - i - 1
			}
		}
	} else {
		//        Q was determined by a call to ZHPTRD with UPLO = 'L'.
		forwrd = (left && !notran) || (!left && notran)

		if forwrd {
			i1 = 1
			i2 = nq - 1
			i3 = 1
			ii = 2
		} else {
			i1 = nq - 1
			i2 = 1
			i3 = -1
			ii = nq*(nq+1)/2 - 1
		}

		if left {
			ni = (*n)
			jc = 1
		} else {
			mi = (*m)
			ic = 1
		}

		for _, i = range genIter(i1, i2, i3) {
			aii = ap.Get(ii - 1)
			ap.Set(ii-1, one)
			if left {
				//              H(i) or H(i)**H is applied to C(i+1:m,1:n)
				mi = (*m) - i
				ic = i + 1
			} else {
				//              H(i) or H(i)**H is applied to C(1:m,i+1:n)
				ni = (*n) - i
				jc = i + 1
			}

			//           Apply H(i) or H(i)**H
			if notran {
				taui = tau.Get(i - 1)
			} else {
				taui = tau.GetConj(i - 1)
			}
			Zlarf(side, &mi, &ni, ap.Off(ii-1), func() *int { y := 1; return &y }(), &taui, c.Off(ic-1, jc-1), ldc, work)
			ap.Set(ii-1, aii)

			if forwrd {
				ii = ii + nq - i + 1
			} else {
				ii = ii - nq + i - 2
			}
		}
	}
}
