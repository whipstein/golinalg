package golapack

import (
	"fmt"

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
func Zgelq(m, n int, a *mat.CMatrix, t *mat.CVector, tsize int, work *mat.CVector, lwork int) (err error) {
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
		mb = Ilaenv(1, "Zgelq", []byte{' '}, m, n, 1, -1)
		nb = Ilaenv(1, "Zgelq", []byte{' '}, m, n, 2, -1)
	} else {
		mb = 1
		nb = n
	}
	if mb > min(m, n) || mb < 1 {
		mb = 1
	}
	if nb > n || nb <= m {
		nb = n
	}
	mintsz = m + 5
	if nb > m && n > m {
		if (n-m)%(nb-m) == 0 {
			nblcks = (n - m) / (nb - m)
		} else {
			nblcks = (n-m)/(nb-m) + 1
		}
	} else {
		nblcks = 1
	}

	//     Determine if the workspace size satisfies minimal size
	lminws = false
	if (tsize < max(1, mb*m*nblcks+5) || lwork < mb*m) && (lwork >= m) && (tsize >= mintsz) && (!lquery) {
		if tsize < max(1, mb*m*nblcks+5) {
			lminws = true
			mb = 1
			nb = n
		}
		if lwork < mb*m {
			lminws = true
			mb = 1
		}
	}

	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if tsize < max(1, mb*m*nblcks+5) && (!lquery) && (!lminws) {
		err = fmt.Errorf("tsize < max(1, mb*m*nblcks+5) && (!lquery) && (!lminws): tsize=%v, mb=%v, m=%v, nblcks=%v, lquery=%v, lminws=%v", tsize, mb, m, nblcks, lquery, lminws)
	} else if (lwork < max(1, m*mb)) && (!lquery) && (!lminws) {
		err = fmt.Errorf("(lwork < max(1, m*mb)) && (!lquery) && (!lminws): lwork=%v, m=%v, mb=%v, lquery=%v, lminws=%v", lwork, m, mb, lquery, lminws)
	}

	if err == nil {
		if mint {
			t.SetRe(0, float64(mintsz))
		} else {
			t.SetRe(0, float64(mb*m*nblcks+5))
		}
		t.SetRe(1, float64(mb))
		t.SetRe(2, float64(nb))
		if minw {
			work.SetRe(0, float64(max(1, n)))
		} else {
			work.SetRe(0, float64(max(1, mb*m)))
		}
	}
	if err != nil {
		gltest.Xerbla2("Zgelq", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if min(m, n) == 0 {
		return
	}

	//     The LQ Decomposition
	if (n <= m) || (nb <= m) || (nb >= n) {
		if err = Zgelqt(m, n, mb, a, t.Off(5).CMatrix(mb, opts), work); err != nil {
			panic(err)
		}
	} else {
		if err = Zlaswlq(m, n, mb, nb, a, t.Off(5).CMatrix(mb, opts), work, lwork); err != nil {
			panic(err)
		}
	}

	work.SetRe(0, float64(max(1, mb*m)))

	return
}
