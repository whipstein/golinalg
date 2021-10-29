package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztplqt computes a blocked LQ factorization of a complex
// "triangular-pentagonal" matrix C, which is composed of a
// triangular block A and pentagonal block B, using the compact
// WY representation for Q.
func Ztplqt(m, n, l, mb int, a, b, t *mat.CMatrix, work *mat.CVector) (err error) {
	var i, ib, lb, nb int

	//     Test the input arguments
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if l < 0 || (l > min(m, n) && min(m, n) >= 0) {
		err = fmt.Errorf("l < 0 || (l > min(m, n) && min(m, n) >= 0): l=%v, m=%v, n=%v", l, m, n)
	} else if mb < 1 || (mb > m && m > 0) {
		err = fmt.Errorf("mb < 1 || (mb > m && m > 0): mb=%v, m=%v", mb, m)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if b.Rows < max(1, m) {
		err = fmt.Errorf("b.Rows < max(1, m): b.Rows=%v, m=%v", b.Rows, m)
	} else if t.Rows < mb {
		err = fmt.Errorf("t.Rows < mb: t.Rows=%v, mb=%v", t.Rows, mb)
	}
	if err != nil {
		gltest.Xerbla2("Ztplqt", err)
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 {
		return
	}

	for i = 1; i <= m; i += mb {
		//     Compute the QR factorization of the current block
		ib = min(m-i+1, mb)
		nb = min(n-l+i+ib-1, n)
		if i >= l {
			lb = 0
		} else {
			lb = nb - n + l - i + 1
		}

		if err = Ztplqt2(ib, nb, lb, a.Off(i-1, i-1), b.Off(i-1, 0), t.Off(0, i-1)); err != nil {
			panic(err)
		}

		//     Update by applying H**T to B(I+IB:M,:) from the right
		if i+ib <= m {
			if err = Ztprfb(Right, NoTrans, 'F', 'R', m-i-ib+1, nb, ib, lb, b.Off(i-1, 0), t.Off(0, i-1), a.Off(i+ib-1, i-1), b.Off(i+ib-1, 0), work.CMatrix(m-i-ib+1, opts)); err != nil {
				panic(err)
			}
		}
	}

	return
}
