package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zunmrz overwrites the general complex M-by-N matrix C with
//
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      Q * C          C * Q
// TRANS = 'C':      Q**H * C       C * Q**H
//
// where Q is a complex unitary matrix defined as the product of k
// elementary reflectors
//
//       Q = H(1) H(2) . . . H(k)
//
// as returned by ZTZRZF. Q is of order M if SIDE = 'L' and of order N
// if SIDE = 'R'.
func Zunmrz(side, trans byte, m, n, k, l *int, a *mat.CMatrix, lda *int, tau *mat.CVector, c *mat.CMatrix, ldc *int, work *mat.CVector, lwork, info *int) {
	var left, lquery, notran bool
	var transt byte
	var i, i1, i2, i3, ib, ic, iinfo, iwt, ja, jc, ldt, ldwork, lwkopt, mi, nb, nbmax, nbmin, ni, nq, nw, tsize int

	nbmax = 64
	ldt = nbmax + 1
	tsize = ldt * nbmax

	//     Test the input arguments
	(*info) = 0
	left = side == 'L'
	notran = trans == 'N'
	lquery = ((*lwork) == -1)

	//     NQ is the order of Q and NW is the minimum dimension of WORK
	if left {
		nq = (*m)
		nw = max(1, *n)
	} else {
		nq = (*n)
		nw = max(1, *m)
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
	} else if (*lda) < max(1, *k) {
		(*info) = -8
	} else if (*ldc) < max(1, *m) {
		(*info) = -11
	} else if (*lwork) < max(1, nw) && !lquery {
		(*info) = -13
	}

	if (*info) == 0 {
		//        Compute the workspace requirements
		if (*m) == 0 || (*n) == 0 {
			lwkopt = 1
		} else {
			nb = min(nbmax, Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMRQ"), []byte{side, trans}, m, n, k, toPtr(-1)))
			lwkopt = nw*nb + tsize
		}
		work.SetRe(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNMRZ"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		return
	}

	//     Determine the block size.  NB may be at most NBMAX, where NBMAX
	//     is used to define the local array T.
	nb = min(nbmax, Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMRQ"), []byte{side, trans}, m, n, k, toPtr(-1)))
	nbmin = 2
	ldwork = nw
	if nb > 1 && nb < (*k) {
		if (*lwork) < nw*nb+tsize {
			nb = ((*lwork) - tsize) / ldwork
			nbmin = max(2, Ilaenv(func() *int { y := 2; return &y }(), []byte("ZUNMRQ"), []byte{side, trans}, m, n, k, toPtr(-1)))
		}
	}

	if nb < nbmin || nb >= (*k) {
		//        Use unblocked code
		Zunmr3(side, trans, m, n, k, l, a, lda, tau, c, ldc, work, &iinfo)
	} else {
		//        Use blocked code
		iwt = 1 + nw*nb
		if (left && !notran) || (!left && notran) {
			i1 = 1
			i2 = (*k)
			i3 = nb
		} else {
			i1 = (((*k)-1)/nb)*nb + 1
			i2 = 1
			i3 = -nb
		}

		if left {
			ni = (*n)
			jc = 1
			ja = (*m) - (*l) + 1
		} else {
			mi = (*m)
			ic = 1
			ja = (*n) - (*l) + 1
		}

		if notran {
			transt = 'C'
		} else {
			transt = 'N'
		}

		for _, i = range genIter(i1, i2, i3) {
			ib = min(nb, (*k)-i+1)

			//           Form the triangular factor of the block reflector
			//           H = H(i+ib-1) . . . H(i+1) H(i)
			Zlarzt('B', 'R', l, &ib, a.Off(i-1, ja-1), lda, tau.Off(i-1), work.CMatrixOff(iwt-1, ldt, opts), &ldt)

			if left {
				//              H or H**H is applied to C(i:m,1:n)
				mi = (*m) - i + 1
				ic = i
			} else {
				//              H or H**H is applied to C(1:m,i:n)
				ni = (*n) - i + 1
				jc = i
			}

			//           Apply H or H**H
			Zlarzb(side, transt, 'B', 'R', &mi, &ni, &ib, l, a.Off(i-1, ja-1), lda, work.CMatrixOff(iwt-1, ldt, opts), &ldt, c.Off(ic-1, jc-1), ldc, work.CMatrix(ldwork, opts), &ldwork)
		}

	}

	work.SetRe(0, float64(lwkopt))
}
