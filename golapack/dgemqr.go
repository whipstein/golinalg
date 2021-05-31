package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgemqr overwrites the general real M-by-N matrix C with
//
//                      SIDE = 'L'     SIDE = 'R'
//      TRANS = 'N':      Q * C          C * Q
//      TRANS = 'T':      Q**T * C       C * Q**T
//
// where Q is a real orthogonal matrix defined as the product
// of blocked elementary reflectors computed by tall skinny
// QR factorization (DGEQR)
func Dgemqr(side, trans byte, m, n, k *int, a *mat.Matrix, lda *int, t *mat.Vector, tsize *int, c *mat.Matrix, ldc *int, work *mat.Vector, lwork, info *int) {
	var left, lquery, notran, right, tran bool
	var lw, mb, mn, nb, nblcks int
	_ = nblcks

	//     Test the input arguments
	lquery = (*lwork) == -1
	notran = trans == 'N'
	tran = trans == 'T'
	left = side == 'L'
	right = side == 'R'

	mb = int(t.Get(1))
	nb = int(t.Get(2))
	if left {
		lw = (*n) * nb
		mn = (*m)
	} else {
		lw = mb * nb
		mn = (*n)
	}

	if (mb > (*k)) && (mn > (*k)) {
		if (mn-(*k))%(mb-(*k)) == 0 {
			nblcks = (mn - (*k)) / (mb - (*k))
		} else {
			nblcks = (mn-(*k))/(mb-(*k)) + 1
		}
	} else {
		nblcks = 1
	}

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
	} else if (*lda) < maxint(1, mn) {
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
		gltest.Xerbla([]byte("DGEMQR"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if minint(*m, *n, *k) == 0 {
		return
	}

	if (left && (*m) <= (*k)) || (right && (*n) <= (*k)) || (mb <= (*k)) || (mb >= maxint(*m, *n, *k)) {
		Dgemqrt(side, trans, m, n, k, &nb, a, lda, t.MatrixOff(5, nb, opts), &nb, c, ldc, work, info)
	} else {
		Dlamtsqr(side, trans, m, n, k, &mb, &nb, a, lda, t.MatrixOff(5, nb, opts), &nb, c, ldc, work, lwork, info)
	}

	work.Set(0, float64(lw))
}
