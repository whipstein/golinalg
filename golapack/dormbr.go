package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dormbr If VECT = 'Q', DORMBR overwrites the general real M-by-N matrix C
// with
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      Q * C          C * Q
// TRANS = 'T':      Q**T * C       C * Q**T
//
// If VECT = 'P', DORMBR overwrites the general real M-by-N matrix C
// with
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      P * C          C * P
// TRANS = 'T':      P**T * C       C * P**T
//
// Here Q and P**T are the orthogonal matrices determined by DGEBRD when
// reducing a real matrix A to bidiagonal form: A = Q * B * P**T. Q and
// P**T are defined as products of elementary reflectors H(i) and G(i)
// respectively.
//
// Let nq = m if SIDE = 'L' and nq = n if SIDE = 'R'. Thus nq is the
// order of the orthogonal matrix Q or P**T that is applied.
//
// If VECT = 'Q', A is assumed to have been an NQ-by-K matrix:
// if nq >= k, Q = H(1) H(2) . . . H(k);
// if nq < k, Q = H(1) H(2) . . . H(nq-1).
//
// If VECT = 'P', A is assumed to have been a K-by-NQ matrix:
// if k < nq, P = G(1) G(2) . . . G(k);
// if k >= nq, P = G(1) G(2) . . . G(nq-1).
func Dormbr(vect, side, trans byte, m, n, k *int, a *mat.Matrix, lda *int, tau *mat.Vector, c *mat.Matrix, ldc *int, work *mat.Vector, lwork, info *int) {
	var applyq, left, lquery, notran bool
	var transt byte
	var i1, i2, iinfo, lwkopt, mi, nb, ni, nq, nw int

	//     Test the input arguments
	(*info) = 0
	applyq = vect == 'Q'
	left = side == 'L'
	notran = trans == 'N'
	lquery = ((*lwork) == -1)

	//     NQ is the order of Q or P and NW is the minimum dimension of WORK
	if left {
		nq = (*m)
		nw = (*n)
	} else {
		nq = (*n)
		nw = (*m)
	}
	if !applyq && vect != 'P' {
		(*info) = -1
	} else if !left && side != 'R' {
		(*info) = -2
	} else if !notran && trans != 'T' {
		(*info) = -3
	} else if (*m) < 0 {
		(*info) = -4
	} else if (*n) < 0 {
		(*info) = -5
	} else if (*k) < 0 {
		(*info) = -6
	} else if (applyq && (*lda) < maxint(1, nq)) || (!applyq && (*lda) < maxint(1, minint(nq, *k))) {
		(*info) = -8
	} else if (*ldc) < maxint(1, *m) {
		(*info) = -11
	} else if (*lwork) < maxint(1, nw) && !lquery {
		(*info) = -13
	}

	if (*info) == 0 {
		if applyq {
			if left {
				nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMQR"), []byte{side, trans}, toPtr((*m)-1), n, toPtr((*m)-1), toPtr(-1))
			} else {
				nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMQR"), []byte{side, trans}, m, toPtr((*n)-1), toPtr((*n)-1), toPtr(-1))
			}
		} else {
			if left {
				nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMLQ"), []byte{side, trans}, toPtr((*m)-1), n, toPtr((*m)-1), toPtr(-1))
			} else {
				nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMLQ"), []byte{side, trans}, m, toPtr((*n)-1), toPtr((*n)-1), toPtr(-1))
			}
		}
		lwkopt = maxint(1, nw) * nb
		work.Set(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DORMBR"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	work.Set(0, 1)
	if (*m) == 0 || (*n) == 0 {
		return
	}

	if applyq {
		//        Apply Q
		if nq >= (*k) {
			//           Q was determined by a call to DGEBRD with nq >= k
			Dormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, &iinfo)
		} else if nq > 1 {
			//           Q was determined by a call to DGEBRD with nq < k
			if left {
				mi = (*m) - 1
				ni = (*n)
				i1 = 2
				i2 = 1
			} else {
				mi = (*m)
				ni = (*n) - 1
				i1 = 1
				i2 = 2
			}
			Dormqr(side, trans, &mi, &ni, toPtr(nq-1), a.Off(1, 0), lda, tau, c.Off(i1-1, i2-1), ldc, work, lwork, &iinfo)
		}
	} else {
		//        Apply P
		if notran {
			transt = 'T'
		} else {
			transt = 'N'
		}
		if nq > (*k) {
			//           P was determined by a call to DGEBRD with nq > k
			Dormlq(side, transt, m, n, k, a, lda, tau, c, ldc, work, lwork, &iinfo)
		} else if nq > 1 {
			//           P was determined by a call to DGEBRD with nq <= k
			if left {
				mi = (*m) - 1
				ni = (*n)
				i1 = 2
				i2 = 1
			} else {
				mi = (*m)
				ni = (*n) - 1
				i1 = 1
				i2 = 2
			}
			Dormlq(side, transt, &mi, &ni, toPtr(nq-1), a.Off(0, 1), lda, tau, c.Off(i1-1, i2-1), ldc, work, lwork, &iinfo)
		}
	}
	work.Set(0, float64(lwkopt))
}
