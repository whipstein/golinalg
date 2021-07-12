package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlamswlq overwrites the general real M-by-N matrix C with
//
//
//                    SIDE = 'L'     SIDE = 'R'
//    TRANS = 'N':      Q * C          C * Q
//    TRANS = 'T':      Q**T * C       C * Q**T
//    where Q is a real orthogonal matrix defined as the product of blocked
//    elementary reflectors computed by short wide LQ
//    factorization (DLASWLQ)
func Dlamswlq(side, trans byte, m, n, k, mb, nb *int, a *mat.Matrix, lda *int, t *mat.Matrix, ldt *int, c *mat.Matrix, ldc *int, work *mat.Vector, lwork, info *int) {
	var left, lquery, notran, right, tran bool
	var ctr, i, ii, kk, lw int

	//     Test the input arguments
	lquery = (*lwork) < 0
	notran = trans == 'N'
	tran = trans == 'T'
	left = side == 'L'
	right = side == 'R'
	if left {
		lw = (*n) * (*mb)
	} else {
		lw = (*m) * (*mb)
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
	} else if (*k) < 0 {
		(*info) = -5
	} else if (*lda) < max(1, *k) {
		(*info) = -9
	} else if (*ldt) < max(1, *mb) {
		(*info) = -11
	} else if (*ldc) < max(1, *m) {
		(*info) = -13
	} else if ((*lwork) < max(1, lw)) && (!lquery) {
		(*info) = -15
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DLAMSWLQ"), -(*info))
		work.Set(0, float64(lw))
		return
	} else if lquery {
		work.Set(0, float64(lw))
		return
	}

	//     Quick return if possible
	if min(*m, *n, *k) == 0 {
		return
	}

	if ((*nb) <= (*k)) || ((*nb) >= max(*m, *n, *k)) {
		Dgemlqt(side, trans, m, n, k, mb, a, lda, t, ldt, c, ldc, work, info)
		return
	}

	if left && tran {
		//         Multiply Q to the last block of C
		kk = ((*m) - (*k)) % ((*nb) - (*k))
		ctr = ((*m) - (*k)) / ((*nb) - (*k))
		if kk > 0 {
			ii = (*m) - kk + 1
			Dtpmlqt('L', 'T', &kk, n, k, func() *int { y := 0; return &y }(), mb, a.Off(0, ii-1), lda, t.Off(0, ctr*(*k)), ldt, c, ldc, c.Off(ii-1, 0), ldc, work, info)
		} else {
			ii = (*m) + 1
		}

		for i = ii - ((*nb) - (*k)); i >= (*nb)+1; i -= ((*nb) - (*k)) {
			//         Multiply Q to the current block of C (1:M,I:I+NB)
			ctr = ctr - 1
			Dtpmlqt('L', 'T', toPtr((*nb)-(*k)), n, k, func() *int { y := 0; return &y }(), mb, a.Off(0, i-1), lda, t.Off(0, ctr*(*k)), ldt, c, ldc, c.Off(i-1, 0), ldc, work, info)
		}

		//         Multiply Q to the first block of C (1:M,1:NB)
		Dgemlqt('L', 'T', nb, n, k, mb, a, lda, t, ldt, c, ldc, work, info)

	} else if left && notran {
		//         Multiply Q to the first block of C
		kk = ((*m) - (*k)) % ((*nb) - (*k))
		ii = (*m) - kk + 1
		ctr = 1
		Dgemlqt('L', 'N', nb, n, k, mb, a, lda, t, ldt, c, ldc, work, info)

		for i = (*nb) + 1; i <= ii-(*nb)+(*k); i += ((*nb) - (*k)) {
			//         Multiply Q to the current block of C (I:I+NB,1:N)
			Dtpmlqt('L', 'N', toPtr((*nb)-(*k)), n, k, func() *int { y := 0; return &y }(), mb, a.Off(0, i-1), lda, t.Off(0, ctr*(*k)), ldt, c, ldc, c.Off(i-1, 0), ldc, work, info)
			ctr = ctr + 1

		}
		if ii <= (*m) {
			//         Multiply Q to the last block of C
			Dtpmlqt('L', 'N', &kk, n, k, func() *int { y := 0; return &y }(), mb, a.Off(0, ii-1), lda, t.Off(0, ctr*(*k)), ldt, c, ldc, c.Off(ii-1, 0), ldc, work, info)

		}

	} else if right && notran {
		//         Multiply Q to the last block of C
		kk = ((*n) - (*k)) % ((*nb) - (*k))
		ctr = ((*n) - (*k)) / ((*nb) - (*k))
		if kk > 0 {
			ii = (*n) - kk + 1
			Dtpmlqt('R', 'N', m, &kk, k, func() *int { y := 0; return &y }(), mb, a.Off(0, ii-1), lda, t.Off(0, ctr*(*k)), ldt, c, ldc, c.Off(0, ii-1), ldc, work, info)
		} else {
			ii = (*n) + 1
		}

		for i = ii - ((*nb) - (*k)); i >= (*nb)+1; i -= ((*nb) - (*k)) {
			//         Multiply Q to the current block of C (1:M,I:I+MB)
			ctr = ctr - 1
			Dtpmlqt('R', 'N', m, toPtr((*nb)-(*k)), k, func() *int { y := 0; return &y }(), mb, a.Off(0, i-1), lda, t.Off(0, ctr*(*k)), ldt, c, ldc, c.Off(0, i-1), ldc, work, info)

		}

		//         Multiply Q to the first block of C (1:M,1:MB)
		Dgemlqt('R', 'N', m, nb, k, mb, a, lda, t, ldt, c, ldc, work, info)

	} else if right && tran {
		//       Multiply Q to the first block of C
		kk = ((*n) - (*k)) % ((*nb) - (*k))
		ctr = 1
		ii = (*n) - kk + 1
		Dgemlqt('R', 'T', m, nb, k, mb, a, lda, t, ldt, c, ldc, work, info)

		for i = (*nb) + 1; i <= ii-(*nb)+(*k); i += ((*nb) - (*k)) {
			//         Multiply Q to the current block of C (1:M,I:I+MB)
			Dtpmlqt('R', 'T', m, toPtr((*nb)-(*k)), k, func() *int { y := 0; return &y }(), mb, a.Off(0, i-1), lda, t.Off(0, ctr*(*k)), ldt, c, ldc, c.Off(0, i-1), ldc, work, info)
			ctr = ctr + 1

		}
		if ii <= (*n) {
			//       Multiply Q to the last block of C
			Dtpmlqt('R', 'T', m, &kk, k, func() *int { y := 0; return &y }(), mb, a.Off(0, ii-1), lda, t.Off(0, ctr*(*k)), ldt, c, ldc, c.Off(0, ii-1), ldc, work, info)

		}

	}

	work.Set(0, float64(lw))
}
