package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dormtr overwrites the general real M-by-N matrix C with
//
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      Q * C          C * Q
// TRANS = 'T':      Q**T * C       C * Q**T
//
// where Q is a real orthogonal matrix of order nq, with nq = m if
// SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of
// nq-1 elementary reflectors, as returned by DSYTRD:
//
// if UPLO = 'U', Q = H(nq-1) . . . H(2) H(1);
//
// if UPLO = 'L', Q = H(1) H(2) . . . H(nq-1).
func Dormtr(side, uplo, trans byte, m, n *int, a *mat.Matrix, lda *int, tau *mat.Vector, c *mat.Matrix, ldc *int, work *mat.Vector, lwork, info *int) {
	var left, lquery, upper bool
	var i1, i2, iinfo, lwkopt, mi, nb, ni, nq, nw int

	//     Test the input arguments
	*info = 0
	left = side == 'L'
	upper = uplo == 'U'
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
	} else if !upper && uplo != 'L' {
		(*info) = -2
	} else if trans != 'N' && trans != 'T' {
		(*info) = -3
	} else if (*m) < 0 {
		(*info) = -4
	} else if (*n) < 0 {
		(*info) = -5
	} else if (*lda) < max(1, nq) {
		(*info) = -7
	} else if (*ldc) < max(1, *m) {
		(*info) = -10
	} else if (*lwork) < max(1, nw) && !lquery {
		(*info) = -12
	}

	if (*info) == 0 {
		if upper {
			if left {
				nb = Ilaenv(toPtr(1), []byte("DORMQL"), []byte{side, trans}, toPtr((*m)-1), n, toPtr((*m)-1), toPtr(-1))
			} else {
				nb = Ilaenv(toPtr(1), []byte("DORMQL"), []byte{side, trans}, m, toPtr((*n)-1), toPtr((*n)-1), toPtr(-1))
			}
		} else {
			if left {
				nb = Ilaenv(toPtr(1), []byte("DORMQR"), []byte{side, trans}, toPtr((*m)-1), n, toPtr((*m)-1), toPtr(-1))
			} else {
				nb = Ilaenv(toPtr(1), []byte("DORMQR"), []byte{side, trans}, m, toPtr((*n)-1), toPtr((*n)-1), toPtr(-1))
			}
		}
		lwkopt = max(1, nw) * nb
		work.Set(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DORMTR"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 || nq == 1 {
		work.Set(0, 1)
		return
	}

	if left {
		mi = (*m) - 1
		ni = (*n)
	} else {
		mi = (*m)
		ni = (*n) - 1
	}

	if upper {
		//        Q was determined by a call to DSYTRD with UPLO = 'U'
		Dormql(side, trans, &mi, &ni, toPtr(nq-1), a.Off(0, 1), lda, tau, c, ldc, work, lwork, &iinfo)
	} else {
		//        Q was determined by a call to DSYTRD with UPLO = 'L'
		if left {
			i1 = 2
			i2 = 1
		} else {
			i1 = 1
			i2 = 2
		}
		Dormqr(side, trans, &mi, &ni, toPtr(nq-1), a.Off(1, 0), lda, tau, c.Off(i1-1, i2-1), ldc, work, lwork, &iinfo)
	}
	work.Set(0, float64(lwkopt))
}
