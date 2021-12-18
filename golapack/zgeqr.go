package golapack

import (
	"fmt"

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
func Zgeqr(m, n int, a *mat.CMatrix, t *mat.CVector, tsize int, work *mat.CVector, lwork int) (err error) {
	var lminws, lquery, mint, minw bool
	var mb, mintsz, nb, nblcks int

	//     Test the input arguments
	lquery = (tsize == -1 || tsize == -2 || lwork == -1 || lwork == -2)

	mint = false
	minw = false
	if tsize == -2 || lwork == -2 {
		if tsize != -1 {
			mint = true
		}
		if lwork != -1 {
			minw = true
		}
	}

	//     Determine the block size
	if min(m, n) > 0 {
		mb = Ilaenv(1, "Zgeqr", []byte{' '}, m, n, 1, -1)
		nb = Ilaenv(1, "Zgeqr", []byte{' '}, m, n, 2, -1)
	} else {
		mb = m
		nb = 1
	}
	if mb > m || mb <= n {
		mb = m
	}
	if nb > min(m, n) || nb < 1 {
		nb = 1
	}
	mintsz = n + 5
	if mb > n && m > n {
		if (m-n)%(mb-n) == 0 {
			nblcks = (m - n) / (mb - n)
		} else {
			nblcks = (m-n)/(mb-n) + 1
		}
	} else {
		nblcks = 1
	}
	//
	//     Determine if the workspace size satisfies minimal size
	//
	lminws = false
	if (tsize < max(1, nb*n*nblcks+5) || lwork < nb*n) && (lwork >= n) && (tsize >= mintsz) && (!lquery) {
		if tsize < max(1, nb*n*nblcks+5) {
			lminws = true
			nb = 1
			mb = m
		}
		if lwork < nb*n {
			lminws = true
			nb = 1
		}
	}

	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if tsize < max(1, nb*n*nblcks+5) && (!lquery) && (!lminws) {
		err = fmt.Errorf("tsize < max(1, nb*n*nblcks+5) && (!lquery) && (!lminws): tsize=%v, nb=%v, n=%v, nblcks=%v, lquery=%v, lminws=%v", tsize, nb, n, nblcks, lquery, lminws)
	} else if (lwork < max(1, n*nb)) && (!lquery) && (!lminws) {
		err = fmt.Errorf("(lwork < max(1, n*nb)) && (!lquery) && (!lminws): lwork=%v, n=%v, nb=%v, lquery=%v, lminws=%v", lwork, n, nb, lquery, lminws)
	}

	if err == nil {
		if mint {
			t.SetRe(0, float64(mintsz))
		} else {
			t.SetRe(0, float64(nb*n*nblcks+5))
		}
		t.SetRe(1, float64(mb))
		t.SetRe(2, float64(nb))
		if minw {
			work.SetRe(0, float64(max(1, n)))
		} else {
			work.SetRe(0, float64(max(1, nb*n)))
		}
	}
	if err != nil {
		gltest.Xerbla2("Zgeqr", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if min(m, n) == 0 {
		return
	}

	//     The QR Decomposition
	if (m <= n) || (mb <= n) || (mb >= m) {
		if err = Zgeqrt(m, n, nb, a, t.Off(5).CMatrix(nb, opts), work); err != nil {
			panic(err)
		}
	} else {
		if err = Zlatsqr(m, n, mb, nb, a, t.Off(5).CMatrix(nb, opts), work, lwork); err != nil {
			panic(err)
		}
	}

	work.SetRe(0, float64(max(1, nb*n)))

	return
}
