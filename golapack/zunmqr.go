package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zunmqr overwrites the general complex M-by-N matrix C with
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
// as returned by ZGEQRF. Q is of order M if SIDE = 'L' and of order N
// if SIDE = 'R'.
func Zunmqr(side, trans byte, m, n, k *int, a *mat.CMatrix, lda *int, tau *mat.CVector, c *mat.CMatrix, ldc *int, work *mat.CVector, lwork, info *int) {
	var left, lquery, notran bool
	var i, i1, i2, i3, ib, ic, iinfo, iwt, jc, ldt, ldwork, lwkopt, mi, nb, nbmax, nbmin, ni, nq, nw, tsize int

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
		nw = (*n)
	} else {
		nq = (*n)
		nw = (*m)
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
	} else if (*lda) < maxint(1, nq) {
		(*info) = -7
	} else if (*ldc) < maxint(1, *m) {
		(*info) = -10
	} else if (*lwork) < maxint(1, nw) && !lquery {
		(*info) = -12
	}

	if (*info) == 0 {
		//        Compute the workspace requirements
		nb = minint(nbmax, Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMQR"), []byte{side, trans}, m, n, k, toPtr(-1)))
		lwkopt = maxint(1, nw)*nb + tsize
		work.SetRe(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNMQR"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 || (*k) == 0 {
		work.Set(0, 1)
		return
	}

	nbmin = 2
	ldwork = nw
	if nb > 1 && nb < (*k) {
		if (*lwork) < nw*nb+tsize {
			nb = ((*lwork) - tsize) / ldwork
			nbmin = maxint(2, Ilaenv(func() *int { y := 2; return &y }(), []byte("ZUNMQR"), []byte{side, trans}, m, n, k, toPtr(-1)))
		}
	}

	if nb < nbmin || nb >= (*k) {
		//        Use unblocked code
		Zunm2r(side, trans, m, n, k, a, lda, tau, c, ldc, work, &iinfo)
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
		} else {
			mi = (*m)
			ic = 1
		}

		for _, i = range genIter(i1, i2, i3) {
			ib = minint(nb, (*k)-i+1)

			//           Form the triangular factor of the block reflector
			//           H = H(i) H(i+1) . . . H(i+ib-1)
			Zlarft('F', 'C', toPtr(nq-i+1), &ib, a.Off(i-1, i-1), lda, tau.Off(i-1), work.CMatrixOff(iwt-1, ldt, opts), &ldt)
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
			Zlarfb(side, trans, 'F', 'C', &mi, &ni, &ib, a.Off(i-1, i-1), lda, work.CMatrixOff(iwt-1, ldt, opts), &ldt, c.Off(ic-1, jc-1), ldc, work.CMatrix(ldwork, opts), &ldwork)
		}
	}
	work.SetRe(0, float64(lwkopt))
}
