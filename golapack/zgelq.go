package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgelq computes an LQ factorization of a complex M-by-N matrix A:
//
//    A = ( L 0 ) *  Q
//
// where:
//
//    Q is a N-by-N orthogonal matrix;
//    L is an lower-triangular M-by-M matrix;
//    0 is a M-by-(N-M) zero matrix, if M < N.
func Zgelq(m, n *int, a *mat.CMatrix, lda *int, t *mat.CVector, tsize *int, work *mat.CVector, lwork, info *int) {
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
	if minint(*m, *n) > 0 {
		mb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGELQ "), []byte{' '}, m, n, func() *int { y := 1; return &y }(), toPtr(-1))
		nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGELQ "), []byte{' '}, m, n, func() *int { y := 2; return &y }(), toPtr(-1))
	} else {
		mb = 1
		nb = (*n)
	}
	if mb > minint(*m, *n) || mb < 1 {
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
	if ((*tsize) < maxint(1, mb*(*m)*nblcks+5) || (*lwork) < mb*(*m)) && ((*lwork) >= (*m)) && ((*tsize) >= mintsz) && (!lquery) {
		if (*tsize) < maxint(1, mb*(*m)*nblcks+5) {
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
	} else if (*lda) < maxint(1, *m) {
		(*info) = -4
	} else if (*tsize) < maxint(1, mb*(*m)*nblcks+5) && (!lquery) && (!lminws) {
		(*info) = -6
	} else if ((*lwork) < maxint(1, (*m)*mb)) && (!lquery) && (!lminws) {
		(*info) = -8
	}

	if (*info) == 0 {
		if mint {
			t.SetRe(0, float64(mintsz))
		} else {
			t.SetRe(0, float64(mb*(*m)*nblcks+5))
		}
		t.SetRe(1, float64(mb))
		t.SetRe(2, float64(nb))
		if minw {
			work.SetRe(0, float64(maxint(1, *n)))
		} else {
			work.SetRe(0, float64(maxint(1, mb*(*m))))
		}
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGELQ"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if minint(*m, *n) == 0 {
		return
	}

	//     The LQ Decomposition
	if ((*n) <= (*m)) || (nb <= (*m)) || (nb >= (*n)) {
		Zgelqt(m, n, &mb, a, lda, t.CMatrixOff(5, mb, opts), &mb, work, info)
	} else {
		Zlaswlq(m, n, &mb, &nb, a, lda, t.CMatrixOff(5, mb, opts), &mb, work, lwork, info)
	}

	work.SetRe(0, float64(maxint(1, mb*(*m))))
}
