package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zlamtsqr overwrites the general complex M-by-N matrix C with
//
//
//                 SIDE = 'L'     SIDE = 'R'
// TRANS = 'N':      Q * C          C * Q
// TRANS = 'C':      Q**H * C       C * Q**H
//      where Q is a real orthogonal matrix defined as the product
//      of blocked elementary reflectors computed by tall skinny
//      QR factorization (ZLATSQR)
func Zlamtsqr(side, trans byte, m, n, k, mb, nb *int, a *mat.CMatrix, lda *int, t *mat.CMatrix, ldt *int, c *mat.CMatrix, ldc *int, work *mat.CVector, lwork, info *int) {
	var left, lquery, notran, right, tran bool
	var ctr, i, ii, kk, lw int

	//     Test the input arguments
	lquery = (*lwork) < 0
	notran = trans == 'N'
	tran = trans == 'C'
	left = side == 'L'
	right = side == 'R'
	if left {
		lw = (*n) * (*nb)
	} else {
		lw = (*m) * (*nb)
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
	} else if (*ldt) < maxint(1, *nb) {
		(*info) = -11
	} else if (*ldc) < maxint(1, *m) {
		(*info) = -13
	} else if ((*lwork) < maxint(1, lw)) && (!lquery) {
		(*info) = -15
	}

	//     Determine the block size if it is tall skinny or short and wide
	if (*info) == 0 {
		work.SetRe(0, float64(lw))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZLAMTSQR"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if minint(*m, *n, *k) == 0 {
		return
	}

	if ((*mb) <= (*k)) || ((*mb) >= maxint(*m, *n, *k)) {
		Zgemqrt(side, trans, m, n, k, nb, a, lda, t, ldt, c, ldc, work, info)
		return
	}

	if left && notran {
		//         Multiply Q to the last block of C
		kk = ((*m) - (*k)) % ((*mb) - (*k))
		ctr = ((*m) - (*k)) / ((*mb) - (*k))
		if kk > 0 {
			ii = (*m) - kk + 1
			Ztpmqrt('L', 'N', &kk, n, k, func() *int { y := 0; return &y }(), nb, a.Off(ii-1, 0), lda, t.Off(0, ctr*(*k)+1-1), ldt, c, ldc, c.Off(ii-1, 0), ldc, work, info)
		} else {
			ii = (*m) + 1
		}

		for i = ii - ((*mb) - (*k)); i >= (*mb)+1; i -= ((*mb) - (*k)) {
			//         Multiply Q to the current block of C (I:I+MB,1:N)
			ctr = ctr - 1
			Ztpmqrt('L', 'N', toPtr((*mb)-(*k)), n, k, func() *int { y := 0; return &y }(), nb, a.Off(i-1, 0), lda, t.Off(0, ctr*(*k)+1-1), ldt, c, ldc, c.Off(i-1, 0), ldc, work, info)
		}

		//         Multiply Q to the first block of C (1:MB,1:N)
		Zgemqrt('L', 'N', mb, n, k, nb, a, lda, t, ldt, c, ldc, work, info)

	} else if left && tran {
		//         Multiply Q to the first block of C
		kk = ((*m) - (*k)) % ((*mb) - (*k))
		ii = (*m) - kk + 1
		ctr = 1
		Zgemqrt('L', 'C', mb, n, k, nb, a, lda, t, ldt, c, ldc, work, info)

		for i = (*mb) + 1; i <= ii-(*mb)+(*k); i += ((*mb) - (*k)) {
			//         Multiply Q to the current block of C (I:I+MB,1:N)
			Ztpmqrt('L', 'C', toPtr((*mb)-(*k)), n, k, func() *int { y := 0; return &y }(), nb, a.Off(i-1, 0), lda, t.Off(0, ctr*(*k)+1-1), ldt, c, ldc, c.Off(i-1, 0), ldc, work, info)
			ctr = ctr + 1

		}
		if ii <= (*m) {
			//         Multiply Q to the last block of C
			Ztpmqrt('L', 'C', &kk, n, k, func() *int { y := 0; return &y }(), nb, a.Off(ii-1, 0), lda, t.Off(0, ctr*(*k)+1-1), ldt, c, ldc, c.Off(ii-1, 0), ldc, work, info)

		}

	} else if right && tran {
		//         Multiply Q to the last block of C
		kk = ((*n) - (*k)) % ((*mb) - (*k))
		ctr = ((*n) - (*k)) / ((*mb) - (*k))
		if kk > 0 {
			ii = (*n) - kk + 1
			Ztpmqrt('R', 'C', m, &kk, k, func() *int { y := 0; return &y }(), nb, a.Off(ii-1, 0), lda, t.Off(0, ctr*(*k)+1-1), ldt, c, ldc, c.Off(0, ii-1), ldc, work, info)
		} else {
			ii = (*n) + 1
		}

		for i = ii - ((*mb) - (*k)); i >= (*mb)+1; i -= ((*mb) - (*k)) {
			//         Multiply Q to the current block of C (1:M,I:I+MB)
			ctr = ctr - 1
			Ztpmqrt('R', 'C', m, toPtr((*mb)-(*k)), k, func() *int { y := 0; return &y }(), nb, a.Off(i-1, 0), lda, t.Off(0, ctr*(*k)+1-1), ldt, c, ldc, c.Off(0, i-1), ldc, work, info)
		}

		//         Multiply Q to the first block of C (1:M,1:MB)
		Zgemqrt('R', 'C', m, mb, k, nb, a, lda, t, ldt, c, ldc, work, info)

	} else if right && notran {
		//         Multiply Q to the first block of C
		kk = ((*n) - (*k)) % ((*mb) - (*k))
		ii = (*n) - kk + 1
		ctr = 1
		Zgemqrt('R', 'N', m, mb, k, nb, a, lda, t, ldt, c, ldc, work, info)

		for i = (*mb) + 1; i <= ii-(*mb)+(*k); i += ((*mb) - (*k)) {
			//         Multiply Q to the current block of C (1:M,I:I+MB)
			Ztpmqrt('R', 'N', m, toPtr((*mb)-(*k)), k, func() *int { y := 0; return &y }(), nb, a.Off(i-1, 0), lda, t.Off(0, ctr*(*k)+1-1), ldt, c, ldc, c.Off(0, i-1), ldc, work, info)
			ctr = ctr + 1

		}
		if ii <= (*n) {
			//         Multiply Q to the last block of C
			Ztpmqrt('R', 'N', m, &kk, k, func() *int { y := 0; return &y }(), nb, a.Off(ii-1, 0), lda, t.Off(0, ctr*(*k)+1-1), ldt, c, ldc, c.Off(0, ii-1), ldc, work, info)

		}

	}

	work.SetRe(0, float64(lw))
}
