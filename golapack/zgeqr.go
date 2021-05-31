package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgeqr computes a QR factorization of a complex M-by-N matrix A:
//
//    A = Q * ( R ),
//            ( 0 )
//
// where:
//
//    Q is a M-by-M orthogonal matrix;
//    R is an upper-triangular N-by-N matrix;
//    0 is a (M-N)-by-N zero matrix, if M > N.
func Zgeqr(m, n *int, a *mat.CMatrix, lda *int, t *mat.CVector, tsize *int, work *mat.CVector, lwork, info *int) {
	var lminws, lquery, mint, minw bool
	var mb, mintsz, nb, nblcks int

	//     Test the input arguments
	(*info) = 0

	lquery = ((*tsize) == -1 || (*tsize) == -2 || (*lwork) == -1 || (*lwork) == -2)
	//
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
		mb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEQR "), []byte{' '}, m, n, func() *int { y := 1; return &y }(), toPtr(-1))
		nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEQR "), []byte{' '}, m, n, func() *int { y := 2; return &y }(), toPtr(-1))
	} else {
		mb = (*m)
		nb = 1
	}
	if mb > (*m) || mb <= (*n) {
		mb = (*m)
	}
	if nb > minint(*m, *n) || nb < 1 {
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
	//
	//     Determine if the workspace size satisfies minimal size
	//
	lminws = false
	if ((*tsize) < maxint(1, nb*(*n)*nblcks+5) || (*lwork) < nb*(*n)) && ((*lwork) >= (*n)) && ((*tsize) >= mintsz) && (!lquery) {
		if (*tsize) < maxint(1, nb*(*n)*nblcks+5) {
			lminws = true
			nb = 1
			mb = (*m)
		}
		if (*lwork) < nb*(*n) {
			lminws = true
			nb = 1
		}
	}
	//
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *m) {
		(*info) = -4
	} else if (*tsize) < maxint(1, nb*(*n)*nblcks+5) && (!lquery) && (!lminws) {
		(*info) = -6
	} else if ((*lwork) < maxint(1, (*n)*nb)) && (!lquery) && (!lminws) {
		(*info) = -8
	}
	//
	if (*info) == 0 {
		if mint {
			t.SetRe(0, float64(mintsz))
		} else {
			t.SetRe(0, float64(nb*(*n)*nblcks+5))
		}
		t.SetRe(1, float64(mb))
		t.SetRe(2, float64(nb))
		if minw {
			work.SetRe(0, float64(maxint(1, *n)))
		} else {
			work.SetRe(0, float64(maxint(1, nb*(*n))))
		}
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGEQR"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if minint(*m, *n) == 0 {
		return
	}

	//     The QR Decomposition
	if ((*m) <= (*n)) || (mb <= (*n)) || (mb >= (*m)) {
		Zgeqrt(m, n, &nb, a, lda, t.CMatrixOff(5, nb, opts), &nb, work, info)
	} else {
		Zlatsqr(m, n, &mb, &nb, a, lda, t.CMatrixOff(5, nb, opts), &nb, work, lwork, info)
	}

	work.SetRe(0, float64(maxint(1, nb*(*n))))
}
