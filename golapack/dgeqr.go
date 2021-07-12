package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgeqr computes a QR factorization of a real M-by-N matrix A:
//
//    A = Q * ( R ),
//            ( 0 )
//
// where:
//
//    Q is a M-by-M orthogonal matrix;
//    R is an upper-triangular N-by-N matrix;
//    0 is a (M-N)-by-N zero matrix, if M > N.
func Dgeqr(m, n *int, a *mat.Matrix, lda *int, t *mat.Vector, tsize *int, work *mat.Vector, lwork *int, info *int) {
	var lminws, lquery, mint, minw bool
	var mb, mintsz, nb, nblcks int

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
		mb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DGEQR "), []byte{' '}, m, n, func() *int { y := 1; return &y }(), toPtr(-1))
		nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DGEQR "), []byte{' '}, m, n, func() *int { y := 2; return &y }(), toPtr(-1))
	} else {
		mb = (*m)
		nb = 1
	}
	if mb > (*m) || mb <= (*n) {
		mb = (*m)
	}
	if nb > min(*m, *n) || nb < 1 {
		nb = 1
	}
	mintsz = (*n) + 5
	if mb > (*n) && (*m) > (*n) {
		if ((*m)-(*n))%(mb-(*n)) == 0 {
			nblcks = ((*m) - (*n)) / (mb - (*n))
		} else {
			nblcks = ((*m)-(*n))/(mb-(*n)) + 1
		}
	} else {
		nblcks = 1
	}

	//     Determine if the workspace size satisfies minimal size
	lminws = false
	if ((*tsize) < max(1, nb*(*n)*nblcks+5) || (*lwork) < nb*(*n)) && ((*lwork) >= (*n)) && ((*tsize) >= mintsz) && (!lquery) {
		if (*tsize) < max(1, nb*(*n)*nblcks+5) {
			lminws = true
			nb = 1
			mb = (*m)
		}
		if (*lwork) < nb*(*n) {
			lminws = true
			nb = 1
		}
	}

	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *m) {
		(*info) = -4
	} else if (*tsize) < max(1, nb*(*n)*nblcks+5) && (!lquery) && (!lminws) {
		(*info) = -6
	} else if ((*lwork) < max(1, (*n)*nb)) && (!lquery) && (!lminws) {
		(*info) = -8
	}

	if (*info) == 0 {
		if mint {
			t.Set(0, float64(mintsz))
		} else {
			t.Set(0, float64(nb*(*n)*nblcks+5))
		}
		t.Set(1, float64(mb))
		t.Set(2, float64(nb))
		if minw {
			work.Set(0, float64(max(1, *n)))
		} else {
			work.Set(0, float64(max(1, nb*(*n))))
		}
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGEQR"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if min(*m, *n) == 0 {
		return
	}

	//     The QR Decomposition
	if ((*m) <= (*n)) || (mb <= (*n)) || (mb >= (*m)) {
		Dgeqrt(m, n, &nb, a, lda, t.MatrixOff(5, nb, opts), &nb, work, info)
	} else {
		Dlatsqr(m, n, &mb, &nb, a, lda, t.MatrixOff(5, nb, opts), &nb, work, lwork, info)
	}

	work.Set(0, float64(max(1, nb*(*n))))
}
