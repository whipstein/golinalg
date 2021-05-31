package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlaswlq computes a blocked Tall-Skinny LQ factorization of
// a real M-by-N matrix A for M <= N:
//
//    A = ( L 0 ) *  Q,
//
// where:
//
//    Q is a n-by-N orthogonal matrix, stored on exit in an implicit
//    form in the elements above the digonal of the array A and in
//    the elemenst of the array T;
//    L is an lower-triangular M-by-M matrix stored on exit in
//    the elements on and below the diagonal of the array A.
//    0 is a M-by-(N-M) zero matrix, if M < N, and is not stored.
func Dlaswlq(m, n, mb, nb *int, a *mat.Matrix, lda *int, t *mat.Matrix, ldt *int, work *mat.Vector, lwork, info *int) {
	var lquery bool
	var ctr, i, ii, kk int

	//     TEST THE INPUT ARGUMENTS
	(*info) = 0

	lquery = ((*lwork) == -1)

	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 || (*n) < (*m) {
		(*info) = -2
	} else if (*mb) < 1 || ((*mb) > (*m) && (*m) > 0) {
		(*info) = -3
	} else if (*nb) <= (*m) {
		(*info) = -4
	} else if (*lda) < maxint(1, *m) {
		(*info) = -5
	} else if (*ldt) < (*mb) {
		(*info) = -8
	} else if ((*lwork) < (*m)*(*mb)) && (!lquery) {
		(*info) = -10
	}
	if (*info) == 0 {
		work.Set(0, float64((*mb)*(*m)))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DLASWLQ"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if minint(*m, *n) == 0 {
		return
	}

	//     The LQ Decomposition
	if ((*m) >= (*n)) || ((*nb) <= (*m)) || ((*nb) >= (*n)) {
		Dgelqt(m, n, mb, a, lda, t, ldt, work, info)
		return
	}

	kk = ((*n) - (*m)) % ((*nb) - (*m))
	ii = (*n) - kk + 1

	//      Compute the LQ factorization of the first block A(1:M,1:NB)
	Dgelqt(m, nb, mb, a, lda, t, ldt, work, info)
	ctr = 1

	for i = (*nb) + 1; i <= ii-(*nb)+(*m); i += ((*nb) - (*m)) {
		//      Compute the QR factorization of the current block A(1:M,I:I+NB-M)
		Dtplqt(m, toPtr((*nb)-(*m)), func() *int { y := 0; return &y }(), mb, a, lda, a.Off(0, i-1), lda, t.Off(0, ctr*(*m)+1-1), ldt, work, info)
		ctr = ctr + 1
	}

	//     Compute the QR factorization of the last block A(1:M,II:N)
	if ii <= (*n) {
		Dtplqt(m, &kk, func() *int { y := 0; return &y }(), mb, a, lda, a.Off(0, ii-1), lda, t.Off(0, ctr*(*m)+1-1), ldt, work, info)
	}

	work.Set(0, float64((*m)*(*mb)))
}
