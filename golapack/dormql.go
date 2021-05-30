package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dormql overwrites the general real M-by-N matrix C with
//
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      Q * C          C * Q
// TRANS = 'T':      Q**T * C       C * Q**T
//
// where Q is a real orthogonal matrix defined as the product of k
// elementary reflectors
//
//       Q = H(k) . . . H(2) H(1)
//
// as returned by DGEQLF. Q is of order M if SIDE = 'L' and of order N
// if SIDE = 'R'.
func Dormql(side, trans byte, m, n, k *int, a *mat.Matrix, lda *int, tau *mat.Vector, c *mat.Matrix, ldc *int, work *mat.Vector, lwork, info *int) {
	var left, lquery, notran bool
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
	} else if !notran && trans != 'T' {
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
	} else if (*lwork) < nw && !lquery {
		(*info) = -12
	}

	if (*info) == 0 {
		//        Compute the workspace requirements
		if (*m) == 0 || (*n) == 0 {
			lwkopt = 1
		} else {
			nb = minint(nbmax, Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMQL"), []byte{side, trans}, m, n, k, toPtr(-1)))
			lwkopt = nw*nb + tsize
		}
		work.Set(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DORMQL"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		return
	}

	nbmin = 2
	ldwork = nw
	if nb > 1 && nb < (*k) {
		if (*lwork) < nw*nb+tsize {
			nb = ((*lwork) - tsize) / ldwork
			nbmin = maxint(2, Ilaenv(func() *int { y := 2; return &y }(), []byte("DORMQL"), []byte{side, trans}, m, n, k, toPtr(-1)))
		}
	}

	if nb < nbmin || nb >= (*k) {
		//        Use unblocked code
		Dorm2l(side, trans, m, n, k, a, lda, tau, c, ldc, work, &iinfo)
	} else {
		//        Use blocked code
		iwt = 1 + nw*nb
		if (left && notran) || (!left && !notran) {
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

		for _, i = range genIter(i1, i2, i3) {
			ib = minint(nb, (*k)-i+1)

			//           Form the triangular factor of the block reflector
			//           H = H(i+ib-1) . . . H(i+1) H(i)
			Dlarft('B', 'C', toPtr(nq-(*k)+i+ib-1), &ib, a.Off(0, i-1), lda, tau.Off(i-1), work.MatrixOff(iwt-1, ldt, opts), &ldt)
			if left {
				//              H or H**T is applied to C(1:m-k+i+ib-1,1:n)
				mi = (*m) - (*k) + i + ib - 1
			} else {
				//              H or H**T is applied to C(1:m,1:n-k+i+ib-1)
				ni = (*n) - (*k) + i + ib - 1
			}

			//           Apply H or H**T
			Dlarfb(side, trans, 'B', 'C', &mi, &ni, &ib, a.Off(0, i-1), lda, work.MatrixOff(iwt-1, ldt, opts), &ldt, c, ldc, work.Matrix(ldwork, opts), &ldwork)
		}
	}
	work.Set(0, float64(lwkopt))
}
