package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dgemlq overwrites the general real M-by-N matrix C with
//
//                    SIDE = 'L'     SIDE = 'R'
//    TRANS = 'N':      Q * C          C * Q
//    TRANS = 'T':      Q**T * C       C * Q**T
//    where Q is a real orthogonal matrix defined as the product
//    of blocked elementary reflectors computed by short wide LQ
//    factorization (DGELQ)
func Dgemlq(side, trans byte, m, n, k *int, a *mat.Matrix, lda *int, t *mat.Vector, tsize *int, c *mat.Matrix, ldc *int, work *mat.Vector, lwork, info *int) {
	var left, lquery, notran, right, tran bool
	var lw, mb, mn, nb int

	//     Test the input arguments
	lquery = (*lwork) == -1
	notran = trans == 'N'
	tran = trans == 'T'
	left = side == 'L'
	right = side == 'R'

	mb = int(t.Get(1))
	nb = int(t.Get(2))
	if left {
		lw = (*n) * mb
		mn = (*m)
	} else {
		lw = (*m) * mb
		mn = (*n)
	}

	// if (nb > (*k)) && (mn > (*k)) {
	// 	if (mn-(*k))%(nb-(*k)) == 0 {
	// 		nblcks = (mn - (*k)) / (nb - (*k))
	// 	} else {
	// 		nblcks = (mn-(*k))/(nb-(*k)) + 1
	// 	}
	// } else {
	// 	nblcks = 1
	// }

	(*info) = 0
	if !left && !right {
		(*info) = -1
	} else if !tran && !notran {
		(*info) = -2
	} else if (*m) < 0 {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*k) < 0 || (*k) > mn {
		(*info) = -5
	} else if (*lda) < maxint(1, *k) {
		(*info) = -7
	} else if (*tsize) < 5 {
		(*info) = -9
	} else if (*ldc) < maxint(1, *m) {
		(*info) = -11
	} else if ((*lwork) < maxint(1, lw)) && (!lquery) {
		(*info) = -13
	}

	if (*info) == 0 {
		work.Set(0, float64(lw))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGEMLQ"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if minint(*m, *n, *k) == 0 {
		return
	}

	if (left && (*m) <= (*k)) || (right && (*n) <= (*k)) || (nb <= (*k)) || (nb >= maxint(*m, *n, *k)) {
		Dgemlqt(side, trans, m, n, k, &mb, a, lda, t.MatrixOff(5, mb, opts), &mb, c, ldc, work, info)
	} else {
		Dlamswlq(side, trans, m, n, k, &mb, &nb, a, lda, t.MatrixOff(5, mb, opts), &mb, c, ldc, work, lwork, info)
	}

	work.Set(0, float64(lw))
}
