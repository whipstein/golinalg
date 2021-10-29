package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlatsqr computes a blocked Tall-Skinny QR factorization of
// a real M-by-N matrix A for M >= N:
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
func Dlatsqr(m, n, mb, nb int, a, t *mat.Matrix, work *mat.Vector, lwork int) (err error) {
	var lquery bool
	var ctr, i, ii, kk int

	//     TEST THE INPUT ARGUMENTS

	lquery = (lwork == -1)

	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 || m < n {
		err = fmt.Errorf("n < 0 || m < n: m=%v, n=%v", m, n)
	} else if mb <= n {
		err = fmt.Errorf("mb <= n: n=%v, mb=%v", n, mb)
	} else if nb < 1 || (nb > n && n > 0) {
		err = fmt.Errorf("nb < 1 || (nb > n && n > 0): n=%v, nb=%v", n, nb)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if t.Rows < nb {
		err = fmt.Errorf("t.Rows < nb: t.Rows=%v, nb=%v", t.Rows, nb)
	} else if lwork < (n*nb) && (!lquery) {
		err = fmt.Errorf("lwork < (n*nb) && (!lquery): lwork=%v, n=%v, nb=%v, lquery=%v", lwork, n, nb, lquery)
	}
	if err == nil {
		work.Set(0, float64(nb*n))
	}
	if err != nil {
		gltest.Xerbla2("Dlatsqr", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if min(m, n) == 0 {
		return
	}

	//     The QR Decomposition
	if (mb <= n) || (mb >= m) {
		if err = Dgeqrt(m, n, nb, a, t, work); err != nil {
			panic(err)
		}
		return
	}

	kk = (m - n) % (mb - n)
	ii = m - kk + 1

	//      Compute the QR factorization of the first block A(1:MB,1:N)
	if err = Dgeqrt(mb, n, nb, a, t, work); err != nil {
		panic(err)
	}

	ctr = 1
	for i = mb + 1; i <= ii-mb+n; i += (mb - n) {
		//      Compute the QR factorization of the current block A(I:I+MB-N,1:N)
		if err = Dtpqrt(mb-n, n, 0, nb, a, a.Off(i-1, 0), t.Off(0, ctr*n), work); err != nil {
			panic(err)
		}
		ctr = ctr + 1
	}

	//      Compute the QR factorization of the last block A(II:M,1:N)
	if ii <= m {
		if err = Dtpqrt(kk, n, 0, nb, a, a.Off(ii-1, 0), t.Off(0, ctr*n), work); err != nil {
			panic(err)
		}
	}

	work.Set(0, float64(n*nb))

	return
}
