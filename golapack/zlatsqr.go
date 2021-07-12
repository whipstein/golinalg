package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlatsqr computes a blocked Tall-Skinny QR factorization of
// a complex M-by-N matrix A for M >= N:
//
//    A = Q * ( R ),
//            ( 0 )
//
// where:
//
//    Q is a M-by-M orthogonal matrix, stored on exit in an implicit
//    form in the elements below the digonal of the array A and in
//    the elemenst of the array T;
//
//    R is an upper-triangular N-by-N matrix, stored on exit in
//    the elements on and above the diagonal of the array A.
//
//    0 is a (M-N)-by-N zero matrix, and is not stored.
func Zlatsqr(m, n, mb, nb *int, a *mat.CMatrix, lda *int, t *mat.CMatrix, ldt *int, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var ctr, i, ii, kk int

	//     TEST THE INPUT ARGUMENTS
	(*info) = 0

	lquery = ((*lwork) == -1)

	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 || (*m) < (*n) {
		(*info) = -2
	} else if (*mb) <= (*n) {
		(*info) = -3
	} else if (*nb) < 1 || ((*nb) > (*n) && (*n) > 0) {
		(*info) = -4
	} else if (*lda) < max(1, *m) {
		(*info) = -5
	} else if (*ldt) < (*nb) {
		(*info) = -8
	} else if (*lwork) < ((*n)*(*nb)) && (!lquery) {
		(*info) = -10
	}
	if (*info) == 0 {
		work.SetRe(0, float64((*nb)*(*n)))
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZLATSQR"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if min(*m, *n) == 0 {
		return
	}

	//     The QR Decomposition
	if ((*mb) <= (*n)) || ((*mb) >= (*m)) {
		Zgeqrt(m, n, nb, a, lda, t, ldt, work, info)
		return
	}
	kk = ((*m) - (*n)) % ((*mb) - (*n))
	ii = (*m) - kk + 1

	//      Compute the QR factorization of the first block A(1:MB,1:N)
	Zgeqrt(mb, n, nb, a, lda, t, ldt, work, info)
	ctr = 1

	for i = (*mb) + 1; i <= ii-(*mb)+(*n); i += ((*mb) - (*n)) {
		//      Compute the QR factorization of the current block A(I:I+MB-N,1:N)
		Ztpqrt(toPtr((*mb)-(*n)), n, func() *int { y := 0; return &y }(), nb, a, lda, a.Off(i-1, 0), lda, t.Off(0, ctr*(*n)), ldt, work, info)
		ctr = ctr + 1
	}

	//      Compute the QR factorization of the last block A(II:M,1:N)
	if ii <= (*m) {
		Ztpqrt(&kk, n, func() *int { y := 0; return &y }(), nb, a, lda, a.Off(ii-1, 0), lda, t.Off(0, ctr*(*n)), ldt, work, info)
	}

	work.SetRe(0, float64((*n)*(*nb)))
}
