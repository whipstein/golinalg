package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgelq computes an LQ factorization of a real M-by-N matrix A:
//
//    A = ( L 0 ) *  Q
//
// where:
//
//    Q is a N-by-N orthogonal matrix;
//    L is an lower-triangular M-by-M matrix;
//    0 is a M-by-(N-M) zero matrix, if M < N.
func Dgelq(m, n *int, a *mat.Matrix, lda *int, t *mat.Vector, tsize *int, work *mat.Vector, lwork *int, info *int) {
	var lminws, lquery, mint, minw bool
	var mb, mintsz, nb, nblcks int

	//     Test the input arguments
	(*info) = 0

	lquery = ((*tsize) == -1 || (*tsize) == -2 || (*lwork) == -1 || (*lwork) == -2)

	mint = false
	minw = false
	if (*tsize) == -2 || (*lwork) == -2 {
		if (*tsize) != -1 {
			mint = true
		}
		if (*lwork) != -1 {
			minw = true
		}
	}

	//     Determine the block size
	if min(*m, *n) > 0 {
		mb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DGELQ "), []byte{' '}, m, n, func() *int { y := 1; return &y }(), toPtr(-1))
		nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DGELQ "), []byte{' '}, m, n, func() *int { y := 2; return &y }(), toPtr(-1))
	} else {
		mb = 1
		nb = (*n)
	}
	if mb > min(*m, *n) || mb < 1 {
		mb = 1
	}
	if nb > (*n) || nb <= (*m) {
		nb = (*n)
	}
	mintsz = (*m) + 5
	if nb > (*m) && (*n) > (*m) {
		if ((*n)-(*m))%(nb-(*m)) == 0 {
			nblcks = ((*n) - (*m)) / (nb - (*m))
		} else {
			nblcks = ((*n)-(*m))/(nb-(*m)) + 1
		}
	} else {
		nblcks = 1
	}

	//     Determine if the workspace size satisfies minimal size
	lminws = false
	if ((*tsize) < max(1, mb*(*m)*nblcks+5) || (*lwork) < mb*(*m)) && ((*lwork) >= (*m)) && ((*tsize) >= mintsz) && (!lquery) {
		if (*tsize) < max(1, mb*(*m)*nblcks+5) {
			lminws = true
			mb = 1
			nb = (*n)
		}
		if (*lwork) < mb*(*m) {
			lminws = true
			mb = 1
		}
	}

	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *m) {
		(*info) = -4
	} else if (*tsize) < max(1, mb*(*m)*nblcks+5) && (!lquery) && (!lminws) {
		(*info) = -6
	} else if ((*lwork) < max(1, (*m)*mb)) && (!lquery) && (!lminws) {
		(*info) = -8
	}

	if (*info) == 0 {
		if mint {
			t.Set(0, float64(mintsz))
		} else {
			t.Set(0, float64(mb*(*m)*nblcks+5))
		}
		t.Set(1, float64(mb))
		t.Set(2, float64(nb))
		if minw {
			work.Set(0, float64(max(1, *n)))
		} else {
			work.Set(0, float64(max(1, mb*(*m))))
		}
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGELQ"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if min(*m, *n) == 0 {
		return
	}

	//     The LQ Decomposition
	if ((*n) <= (*m)) || (nb <= (*m)) || (nb >= (*n)) {
		Dgelqt(m, n, &mb, a, lda, t.MatrixOff(5, mb, opts), &mb, work, info)
	} else {
		Dlaswlq(m, n, &mb, &nb, a, lda, t.MatrixOff(5, mb, opts), &mb, work, lwork, info)
	}

	work.Set(0, float64(max(1, mb*(*m))))
}
