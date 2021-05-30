package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zlamswlq overwrites the general real M-by-N matrix C with
//
//
//                    SIDE = 'L'     SIDE = 'R'
//    TRANS = 'N':      Q * C          C * Q
//    TRANS = 'C':      Q**H * C       C * Q**H
//    where Q is a real orthogonal matrix defined as the product of blocked
//    elementary reflectors computed by short wide LQ
//    factorization (ZLASWLQ)
func Zlamswlq(side, trans byte, m, n, k, mb, nb *int, a *mat.CMatrix, lda *int, t *mat.CMatrix, ldt *int, c *mat.CMatrix, ldc *int, work *mat.CVector, lwork, info *int) {
	var left, lquery, notran, right, tran bool
	var ctr, i, ii, kk, lw int

	//     Test the input arguments
	lquery = (*lwork) < 0
	notran = trans == 'N'
	tran = trans == 'C'
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
	} else if (*lda) < maxint(1, *k) {
		(*info) = -9
	} else if (*ldt) < maxint(1, *mb) {
		(*info) = -11
	} else if (*ldc) < maxint(1, *m) {
		(*info) = -13
	} else if ((*lwork) < maxint(1, lw)) && (!lquery) {
		(*info) = -15
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZLAMSWLQ"), -(*info))
		work.SetRe(0, float64(lw))
		return
	} else if lquery {
		work.SetRe(0, float64(lw))
		return
	}

	//     Quick return if possible
	if minint(*m, *n, *k) == 0 {
		return
	}

	if ((*nb) <= (*k)) || ((*nb) >= maxint(*m, *n, *k)) {
		Zgemlqt(side, trans, m, n, k, mb, a, lda, t, ldt, c, ldc, work, info)
		return
	}

	if left && tran {
		//         Multiply Q to the last block of C
		kk = ((*m) - (*k)) % ((*nb) - (*k))
		ctr = ((*m) - (*k)) / ((*nb) - (*k))

		if kk > 0 {
			ii = (*m) - kk + 1
			Ztpmlqt('L', 'C', &kk, n, k, func() *int { y := 0; return &y }(), mb, a.Off(0, ii-1), lda, t.Off(0, ctr*(*k)+1-1), ldt, c, ldc, c.Off(ii-1, 0), ldc, work, info)
		} else {
			ii = (*m) + 1
		}

		for i = ii - ((*nb) - (*k)); i >= (*nb)+1; i -= ((*nb) - (*k)) {
			//         Multiply Q to the current block of C (1:M,I:I+NB)
			ctr = ctr - 1
			Ztpmlqt('L', 'C', toPtr((*nb)-(*k)), n, k, func() *int { y := 0; return &y }(), mb, a.Off(0, i-1), lda, t.Off(0, ctr*(*k)+1-1), ldt, c, ldc, c.Off(i-1, 0), ldc, work, info)
		}

		//         Multiply Q to the first block of C (1:M,1:NB)
		Zgemlqt('L', 'C', nb, n, k, mb, a, lda, t, ldt, c, ldc, work, info)

	} else if left && notran {
		//         Multiply Q to the first block of C
		kk = ((*m) - (*k)) % ((*nb) - (*k))
		ii = (*m) - kk + 1
		ctr = 1
		Zgemlqt('L', 'N', nb, n, k, mb, a, lda, t, ldt, c, ldc, work, info)

		for i = (*nb) + 1; i <= ii-(*nb)+(*k); i += ((*nb) - (*k)) {
			//         Multiply Q to the current block of C (I:I+NB,1:N)
			Ztpmlqt('L', 'N', toPtr((*nb)-(*k)), n, k, func() *int { y := 0; return &y }(), mb, a.Off(0, i-1), lda, t.Off(0, ctr*(*k)+1-1), ldt, c, ldc, c.Off(i-1, 0), ldc, work, info)
			ctr = ctr + 1

		}
		if ii <= (*m) {
			//         Multiply Q to the last block of C
			Ztpmlqt('L', 'N', &kk, n, k, func() *int { y := 0; return &y }(), mb, a.Off(0, ii-1), lda, t.Off(0, ctr*(*k)+1-1), ldt, c, ldc, c.Off(ii-1, 0), ldc, work, info)

		}

	} else if right && notran {
		//         Multiply Q to the last block of C
		kk = ((*n) - (*k)) % ((*nb) - (*k))
		ctr = ((*n) - (*k)) / ((*nb) - (*k))
		if kk > 0 {
			ii = (*n) - kk + 1
			Ztpmlqt('R', 'N', m, &kk, k, func() *int { y := 0; return &y }(), mb, a.Off(0, ii-1), lda, t.Off(0, ctr*(*k)+1-1), ldt, c, ldc, c.Off(0, ii-1), ldc, work, info)
		} else {
			ii = (*n) + 1
		}

		for i = ii - ((*nb) - (*k)); i >= (*nb)+1; i -= ((*nb) - (*k)) {
			//         Multiply Q to the current block of C (1:M,I:I+MB)
			ctr = ctr - 1
			Ztpmlqt('R', 'N', m, toPtr((*nb)-(*k)), k, func() *int { y := 0; return &y }(), mb, a.Off(0, i-1), lda, t.Off(0, ctr*(*k)+1-1), ldt, c, ldc, c.Off(0, i-1), ldc, work, info)
		}

		//         Multiply Q to the first block of C (1:M,1:MB)
		Zgemlqt('R', 'N', m, nb, k, mb, a, lda, t, ldt, c, ldc, work, info)

	} else if right && tran {
		//       Multiply Q to the first block of C
		kk = ((*n) - (*k)) % ((*nb) - (*k))
		ii = (*n) - kk + 1
		Zgemlqt('R', 'C', m, nb, k, mb, a, lda, t, ldt, c, ldc, work, info)
		ctr = 1

		for i = (*nb) + 1; i <= ii-(*nb)+(*k); i += ((*nb) - (*k)) {
			//         Multiply Q to the current block of C (1:M,I:I+MB)
			Ztpmlqt('R', 'C', m, toPtr((*nb)-(*k)), k, func() *int { y := 0; return &y }(), mb, a.Off(0, i-1), lda, t.Off(0, ctr*(*k)+1-1), ldt, c, ldc, c.Off(0, i-1), ldc, work, info)
			ctr = ctr + 1

		}
		if ii <= (*n) {
			//       Multiply Q to the last block of C
			Ztpmlqt('R', 'C', m, &kk, k, func() *int { y := 0; return &y }(), mb, a.Off(0, ii-1), lda, t.Off(0, ctr*(*k)+1-1), ldt, c, ldc, c.Off(0, ii-1), ldc, work, info)

		}

	}

	work.SetRe(0, float64(lw))
}
