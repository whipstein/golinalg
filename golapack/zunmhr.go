package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zunmhr overwrites the general complex M-by-N matrix C with
//
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      Q * C          C * Q
// TRANS = 'C':      Q**H * C       C * Q**H
//
// where Q is a complex unitary matrix of order nq, with nq = m if
// SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of
// IHI-ILO elementary reflectors, as returned by ZGEHRD:
//
// Q = H(ilo) H(ilo+1) . . . H(ihi-1).
func Zunmhr(side, trans byte, m, n, ilo, ihi *int, a *mat.CMatrix, lda *int, tau *mat.CVector, c *mat.CMatrix, ldc *int, work *mat.CVector, lwork, info *int) {
	var left, lquery bool
	var i1, i2, iinfo, lwkopt, mi, nb, nh, ni, nq, nw int

	//     Test the input arguments
	(*info) = 0
	nh = (*ihi) - (*ilo)
	left = side == 'L'
	lquery = ((*lwork) == -1)

	//     NQ is the order of Q and NW is the minimum dimension of WORK
	if left {
		nq = (*m)
		nw = (*n)
	} else {
		nq = (*n)
		nw = (*m)
	}
	if !left && side != 'R' {
		(*info) = -1
	} else if trans != 'N' && trans != 'C' {
		(*info) = -2
	} else if (*m) < 0 {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*ilo) < 1 || (*ilo) > maxint(1, nq) {
		(*info) = -5
	} else if (*ihi) < minint(*ilo, nq) || (*ihi) > nq {
		(*info) = -6
	} else if (*lda) < maxint(1, nq) {
		(*info) = -8
	} else if (*ldc) < maxint(1, *m) {
		(*info) = -11
	} else if (*lwork) < maxint(1, nw) && !lquery {
		(*info) = -13
	}

	if (*info) == 0 {
		if left {
			nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMQR"), []byte{side, trans}, &nh, n, &nh, toPtr(-1))
		} else {
			nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMQR"), []byte{side, trans}, m, &nh, &nh, toPtr(-1))
		}
		lwkopt = maxint(1, nw) * nb
		work.SetRe(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNMHR"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 || nh == 0 {
		work.Set(0, 1)
		return
	}

	if left {
		mi = nh
		ni = (*n)
		i1 = (*ilo) + 1
		i2 = 1
	} else {
		mi = (*m)
		ni = nh
		i1 = 1
		i2 = (*ilo) + 1
	}

	Zunmqr(side, trans, &mi, &ni, &nh, a.Off((*ilo)+1-1, (*ilo)-1), lda, tau.Off((*ilo)-1), c.Off(i1-1, i2-1), ldc, work, lwork, &iinfo)

	work.SetRe(0, float64(lwkopt))
}
