package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zunmrq overwrites the general complex M-by-N matrix C with
//
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      Q * C          C * Q
// TRANS = 'C':      Q**H * C       C * Q**H
//
// where Q is a complex unitary matrix defined as the product of k
// elementary reflectors
//
//       Q = H(1)**H H(2)**H . . . H(k)**H
//
// as returned by ZGERQF. Q is of order M if SIDE = 'L' and of order N
// if SIDE = 'R'.
func Zunmrq(side, trans byte, m, n, k *int, a *mat.CMatrix, lda *int, tau *mat.CVector, c *mat.CMatrix, ldc *int, work *mat.CVector, lwork, info *int) {
	var left, lquery, notran bool
	var transt byte
	var i, i1, i2, i3, ib, iinfo, iwt, ldt, ldwork, lwkopt, mi, nb, nbmax, nbmin, ni, nq, nw, tsize int

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
		nw = maxint(1, *n)
	} else {
		nq = (*n)
		nw = maxint(1, *m)
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
	} else if (*lda) < maxint(1, *k) {
		(*info) = -7
	} else if (*ldc) < maxint(1, *m) {
		(*info) = -10
	} else if (*lwork) < nw && !lquery {
		(*info) = -12
	}

	if (*info) == 0 {
		//        Compute the workspace requirements
		if (*m) == 0 || (*n) == 0 {
			lwkopt = 1
		} else {
			nb = minint(nbmax, Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMRQ"), []byte{side, trans}, m, n, k, toPtr(-1)))
			lwkopt = nw*nb + tsize
		}
		work.SetRe(0, float64(lwkopt))
	}
	//
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNMRQ"), -(*info))
		return
	} else if lquery {
		return
	}
	//
	//     Quick return if possible
	//
	if (*m) == 0 || (*n) == 0 {
		return
	}
	//
	nbmin = 2
	ldwork = nw
	if nb > 1 && nb < (*k) {
		if (*lwork) < nw*nb+tsize {
			nb = ((*lwork) - tsize) / ldwork
			nbmin = maxint(2, Ilaenv(func() *int { y := 2; return &y }(), []byte("ZUNMRQ"), []byte{side, trans}, m, n, k, toPtr(-1)))
		}
	}

	if nb < nbmin || nb >= (*k) {
		//        Use unblocked code
		Zunmr2(side, trans, m, n, k, a, lda, tau, c, ldc, work, &iinfo)
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
		} else {
			mi = (*m)
		}

		if notran {
			transt = 'C'
		} else {
			transt = 'N'
		}

		for _, i = range genIter(i1, i2, i3) {
			ib = minint(nb, (*k)-i+1)

			//           Form the triangular factor of the block reflector
			//           H = H(i+ib-1) . . . H(i+1) H(i)
			Zlarft('B', 'R', toPtr(nq-(*k)+i+ib-1), &ib, a.Off(i-1, 0), lda, tau.Off(i-1), work.CMatrixOff(iwt-1, ldt, opts), &ldt)
			if left {
				//              H or H**H is applied to C(1:m-k+i+ib-1,1:n)
				mi = (*m) - (*k) + i + ib - 1
			} else {
				//              H or H**H is applied to C(1:m,1:n-k+i+ib-1)
				ni = (*n) - (*k) + i + ib - 1
			}

			//           Apply H or H**H
			Zlarfb(side, transt, 'B', 'R', &mi, &ni, &ib, a.Off(i-1, 0), lda, work.CMatrixOff(iwt-1, ldt, opts), &ldt, c, ldc, work.CMatrix(ldwork, opts), &ldwork)
		}
	}
	work.SetRe(0, float64(lwkopt))
}
