package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgeqrt computes a blocked QR factorization of a complex M-by-N matrix A
// using the compact WY representation of Q.
func Zgeqrt(m, n, nb int, a, t *mat.CMatrix, work *mat.CVector) (err error) {
	var useRecursiveQr bool
	var i, ib, k int

	useRecursiveQr = true

	//     Test the input arguments
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nb < 1 || (nb > min(m, n) && min(m, n) > 0) {
		err = fmt.Errorf("nb < 1 || (nb > min(m, n) && min(m, n) > 0): nb=%v, m=%v, n=%v", nb, m, n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if t.Rows < nb {
		err = fmt.Errorf("t.Rows < nb: t.Rows=%v, nb=%v", t.Rows, nb)
	}
	if err != nil {
		gltest.Xerbla2("Zgeqrt", err)
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
			if err = Zgeqrt3(m-i+1, ib, a.Off(i-1, i-1), t.Off(0, i-1)); err != nil {
				panic(err)
			}
		} else {
			if err = Zgeqrt2(m-i+1, ib, a.Off(i-1, i-1), t.Off(0, i-1)); err != nil {
				panic(err)
			}
		}
		if i+ib <= n {
			//     Update by applying H**H to A(I:M,I+IB:N) from the left
			Zlarfb(Left, ConjTrans, 'F', 'C', m-i+1, n-i-ib+1, ib, a.Off(i-1, i-1), t.Off(0, i-1), a.Off(i-1, i+ib-1), work.CMatrix(n-i-ib+1, opts))
		}
	}

	return
}
