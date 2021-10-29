package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtpqrt computes a blocked QR factorization of a real
// "triangular-pentagonal" matrix C, which is composed of a
// triangular block A and pentagonal block B, using the compact
// WY representation for Q.
func Dtpqrt(m, n, l, nb int, a, b, t *mat.Matrix, work *mat.Vector) (err error) {
	var i, ib, lb, mb int

	//     Test the input arguments
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if l < 0 || (l > min(m, n) && min(m, n) >= 0) {
		err = fmt.Errorf("l < 0 || (l > min(m, n) && min(m, n) >= 0): l=%v, m=%v, n=%v", l, m, n)
	} else if nb < 1 || (nb > n && n > 0) {
		err = fmt.Errorf("nb < 1 || (nb > n && n > 0): nb=%v, n=%v", nb, n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, m) {
		err = fmt.Errorf("b.Rows < max(1, m): b.Rows=%v, m=%v", b.Rows, m)
	} else if t.Rows < nb {
		err = fmt.Errorf("t.Rows < nb: t.Rows=%v, nb=%v", t.Rows, nb)
	}
	if err != nil {
		gltest.Xerbla2("Dtpqrt", err)
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 {
		return
	}

	for i = 1; i <= n; i += nb {
		//     Compute the QR factorization of the current block
		ib = min(n-i+1, nb)
		mb = min(m-l+i+ib-1, m)
		if i >= l {
			lb = 0
		} else {
			lb = mb - m + l - i + 1
		}

		if err = Dtpqrt2(mb, ib, lb, a.Off(i-1, i-1), b.Off(0, i-1), t.Off(0, i-1)); err != nil {
			panic(err)
		}

		//     Update by applying H**T to B(:,I+IB:N) from the left
		if i+ib <= n {
			Dtprfb(Left, Trans, 'F', 'C', mb, n-i-ib+1, ib, lb, b.Off(0, i-1), t.Off(0, i-1), a.Off(i-1, i+ib-1), b.Off(0, i+ib-1), work.Matrix(ib, opts))
		}
	}

	return
}
