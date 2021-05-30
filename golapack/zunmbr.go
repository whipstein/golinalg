package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zunmbr If VECT = 'Q', ZUNMBR overwrites the general complex M-by-N matrix C
// with
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      Q * C          C * Q
// TRANS = 'C':      Q**H * C       C * Q**H
//
// If VECT = 'P', ZUNMBR overwrites the general complex M-by-N matrix C
// with
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      P * C          C * P
// TRANS = 'C':      P**H * C       C * P**H
//
// Here Q and P**H are the unitary matrices determined by ZGEBRD when
// reducing a complex matrix A to bidiagonal form: A = Q * B * P**H. Q
// and P**H are defined as products of elementary reflectors H(i) and
// G(i) respectively.
//
// Let nq = m if SIDE = 'L' and nq = n if SIDE = 'R'. Thus nq is the
// order of the unitary matrix Q or P**H that is applied.
//
// If VECT = 'Q', A is assumed to have been an NQ-by-K matrix:
// if nq >= k, Q = H(1) H(2) . . . H(k);
// if nq < k, Q = H(1) H(2) . . . H(nq-1).
//
// If VECT = 'P', A is assumed to have been a K-by-NQ matrix:
// if k < nq, P = G(1) G(2) . . . G(k);
// if k >= nq, P = G(1) G(2) . . . G(nq-1).
func Zunmbr(vect, side, trans byte, m, n, k *int, a *mat.CMatrix, lda *int, tau *mat.CVector, c *mat.CMatrix, ldc *int, work *mat.CVector, lwork, info *int) {
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
	if (*m) == 0 || (*n) == 0 {
		nw = 0
	}
	if !applyq && vect != 'P' {
		(*info) = -1
	} else if !left && side != 'R' {
		(*info) = -2
	} else if !notran && trans != 'C' {
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
		if nw > 0 {
			if applyq {
				if left {
					nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMQR"), []byte{side, trans}, toPtr((*m)-1), n, toPtr((*m)-1), toPtr(-1))
				} else {
					nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMQR"), []byte{side, trans}, m, toPtr((*n)-1), toPtr((*n)-1), toPtr(-1))
				}
			} else {
				if left {
					nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMLQ"), []byte{side, trans}, toPtr((*m)-1), n, toPtr((*m)-1), toPtr(-1))
				} else {
					nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMLQ"), []byte{side, trans}, m, toPtr((*n)-1), toPtr((*n)-1), toPtr(-1))
				}
			}
			lwkopt = maxint(1, nw*nb)
		} else {
			lwkopt = 1
		}
		work.SetRe(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNMBR"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		return
	}

	if applyq {
		//        Apply Q
		if nq >= (*k) {
			//           Q was determined by a call to ZGEBRD with nq >= k
			Zunmqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, &iinfo)
		} else if nq > 1 {
			//           Q was determined by a call to ZGEBRD with nq < k
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
			Zunmqr(side, trans, &mi, &ni, toPtr(nq-1), a.Off(1, 0), lda, tau, c.Off(i1-1, i2-1), ldc, work, lwork, &iinfo)
		}
	} else {
		//        Apply P
		if notran {
			transt = 'C'
		} else {
			transt = 'N'
		}
		if nq > (*k) {
			//           P was determined by a call to ZGEBRD with nq > k
			Zunmlq(side, transt, m, n, k, a, lda, tau, c, ldc, work, lwork, &iinfo)
		} else if nq > 1 {
			//           P was determined by a call to ZGEBRD with nq <= k
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
			Zunmlq(side, transt, &mi, &ni, toPtr(nq-1), a.Off(0, 1), lda, tau, c.Off(i1-1, i2-1), ldc, work, lwork, &iinfo)
		}
	}
	work.SetRe(0, float64(lwkopt))
}
