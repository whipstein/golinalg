package golapack

import (
	"fmt"

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
func Dlaswlq(m, n, mb, nb int, a, t *mat.Matrix, work *mat.Vector, lwork int) (err error) {
	var lquery bool
	var ctr, i, ii, kk int

	//     TEST THE INPUT ARGUMENTS
	lquery = (lwork == -1)

	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 || n < m {
		err = fmt.Errorf("n < 0 || n < m: m=%v, n=%v", m, n)
	} else if mb < 1 || (mb > m && m > 0) {
		err = fmt.Errorf("mb < 1 || (mb > m && m > 0): m=%v, n=%v, mb=%v", m, n, mb)
	} else if nb <= m {
		err = fmt.Errorf("nb <= m: m=%v, nb=%v", m, nb)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if t.Rows < mb {
		err = fmt.Errorf("t.Rows < mb: t.Rows=%v, mb=%v", t.Rows, mb)
	} else if (lwork < m*mb) && (!lquery) {
		err = fmt.Errorf("")
	}
	if err == nil {
		work.Set(0, float64(mb*m))
	}

	if err != nil {
		gltest.Xerbla2("Dlaswlq", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if min(m, n) == 0 {
		return
	}

	//     The LQ Decomposition
	if (m >= n) || (nb <= m) || (nb >= n) {
		if err = Dgelqt(m, n, mb, a, t, work); err != nil {
			panic(err)
		}
		return
	}

	kk = (n - m) % (nb - m)
	ii = n - kk + 1

	//      Compute the LQ factorization of the first block A(1:M,1:NB)
	if err = Dgelqt(m, nb, mb, a, t, work); err != nil {
		panic(err)
	}
	ctr = 1

	for i = nb + 1; i <= ii-nb+m; i += (nb - m) {
		//      Compute the QR factorization of the current block A(1:M,I:I+NB-M)
		if err = Dtplqt(m, nb-m, 0, mb, a, a.Off(0, i-1), t.Off(0, ctr*m), work); err != nil {
			panic(err)
		}
		ctr = ctr + 1
	}

	//     Compute the QR factorization of the last block A(1:M,II:N)
	if ii <= n {
		if err = Dtplqt(m, kk, 0, mb, a, a.Off(0, ii-1), t.Off(0, ctr*m), work); err != nil {
			panic(err)
		}
	}

	work.Set(0, float64(m*mb))

	return
}
