package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgeqrt computes a blocked QR factorization of a real M-by-N matrix A
// using the compact WY representation of Q.
func Dgeqrt(m, n, nb int, a, t *mat.Matrix, work *mat.Vector) (err error) {
	var useRecursiveQr bool
	var i, ib, k int

	useRecursiveQr = true

	//     Test the input arguments
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nb < 1 || (nb > min(m, n) && min(m, n) > 0) {
		err = fmt.Errorf("nb < 1 || (nb > min(m, n) && min(m, n) > 0): m=%v, n=%v, nb=%v", m, n, nb)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if t.Rows < nb {
		err = fmt.Errorf("t.Rows < nb: t.Rows=%v, nb=%v", t.Rows, nb)
	}
	if err != nil {
		gltest.Xerbla2("Dgeqrt", err)
		return
	}

	//     Quick return if possible
	k = min(m, n)
	if k == 0 {
		return
	}

	//     Blocked loop of length K
	for i = 1; i <= k; i += nb {
		ib = min(k-i+1, nb)

		//     Compute the QR factorization of the current block A(I:M,I:I+IB-1)
		if useRecursiveQr {
			if err = Dgeqrt3(m-i+1, ib, a.Off(i-1, i-1), t.Off(0, i-1)); err != nil {
				panic(err)
			}
		} else {
			if err = Dgeqrt2(m-i+1, ib, a.Off(i-1, i-1), t.Off(0, i-1)); err != nil {
				panic(err)
			}
		}
		if i+ib <= n {
			//     Update by applying H**T to A(I:M,I+IB:N) from the left
			Dlarfb(Left, Trans, 'F', 'C', m-i+1, n-i-ib+1, ib, a.Off(i-1, i-1), t.Off(0, i-1), a.Off(i-1, i+ib-1), work.Matrix(n-i-ib+1, opts))
		}
	}

	return
}
