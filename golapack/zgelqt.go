package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgelqt computes a blocked LQ factorization of a complex M-by-N matrix A
// using the compact WY representation of Q.
func Zgelqt(m, n, mb int, a, t *mat.CMatrix, work *mat.CVector) (err error) {
	var i, ib, k int

	//     Test the input arguments
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if mb < 1 || (mb > min(m, n) && min(m, n) > 0) {
		err = fmt.Errorf("mb < 1 || (mb > min(m, n) && min(m, n) > 0): mb=%v, m=%v, n=%v", mb, m, n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if t.Rows < mb {
		err = fmt.Errorf("t.Rows < mb: t.Rows=%v, mb=%v", t.Rows, mb)
	}
	if err != nil {
		gltest.Xerbla2("Zgelqt", err)
		return
	}

	//     Quick return if possible
	k = min(m, n)
	if k == 0 {
		return
	}

	//     Blocked loop of length K
	for i = 1; i <= k; i += mb {
		ib = min(k-i+1, mb)

		//     Compute the LQ factorization of the current block A(I:M,I:I+IB-1)
		if err = Zgelqt3(ib, n-i+1, a.Off(i-1, i-1), t.Off(0, i-1)); err != nil {
			panic(err)
		}
		if i+ib <= m {
			//     Update by applying H**T to A(I:M,I+IB:N) from the right
			Zlarfb(Right, NoTrans, 'F', 'R', m-i-ib+1, n-i+1, ib, a.Off(i-1, i-1), t.Off(0, i-1), a.Off(i+ib-1, i-1), work.CMatrix(m-i-ib+1, opts))
		}
	}

	return
}
