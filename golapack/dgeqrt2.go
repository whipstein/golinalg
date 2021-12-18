package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgeqrt2 computes a QR factorization of a real M-by-N matrix A,
// using the compact WY representation of Q.
func Dgeqrt2(m, n int, a, t *mat.Matrix) (err error) {
	var aii, alpha, one, zero float64
	var i, k int

	one = 1.0e+00
	zero = 0.0e+00

	//     Test the input arguments
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if t.Rows < max(1, n) {
		err = fmt.Errorf("t.Rows < max(1, n): t.Rows=%v, n=%v", t.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dgeqrt2", err)
		return
	}

	k = min(m, n)

	for i = 1; i <= k; i++ {
		//        Generate elem. refl. H(i) to annihilate A(i+1:m,i), tau(I) -> T(I,1)
		*a.GetPtr(i-1, i-1), *t.GetPtr(i-1, 0) = Dlarfg(m-i+1, a.Get(i-1, i-1), a.Off(min(i+1, m)-1, i-1).Vector(), 1)
		if i < n {
			//           Apply H(i) to A(I:M,I+1:N) from the left
			aii = a.Get(i-1, i-1)
			a.Set(i-1, i-1, one)

			//           W(1:N-I) := A(I:M,I+1:N)^H * A(I:M,I) [W = T(:,N)]
			if err = t.Off(0, n-1).Vector().Gemv(Trans, m-i+1, n-i, one, a.Off(i-1, i), a.Off(i-1, i-1).Vector(), 1, zero, 1); err != nil {
				panic(err)
			}

			//           A(I:M,I+1:N) = A(I:m,I+1:N) + alpha*A(I:M,I)*W(1:N-1)^H
			alpha = -t.Get(i-1, 0)
			if err = a.Off(i-1, i).Ger(m-i+1, n-i, alpha, a.Off(i-1, i-1).Vector(), 1, t.Off(0, n-1).Vector(), 1); err != nil {
				panic(err)
			}
			a.Set(i-1, i-1, aii)
		}
	}

	for i = 2; i <= n; i++ {
		aii = a.Get(i-1, i-1)
		a.Set(i-1, i-1, one)

		//        T(1:I-1,I) := alpha * A(I:M,1:I-1)**T * A(I:M,I)
		alpha = -t.Get(i-1, 0)
		if err = t.Off(0, i-1).Vector().Gemv(Trans, m-i+1, i-1, alpha, a.Off(i-1, 0), a.Off(i-1, i-1).Vector(), 1, zero, 1); err != nil {
			panic(err)
		}
		a.Set(i-1, i-1, aii)

		//        T(1:I-1,I) := T(1:I-1,1:I-1) * T(1:I-1,I)
		if err = t.Off(0, i-1).Vector().Trmv(Upper, NoTrans, NonUnit, i-1, t, 1); err != nil {
			panic(err)
		}

		//           T(I,I) = tau(I)
		t.Set(i-1, i-1, t.Get(i-1, 0))
		t.Set(i-1, 0, zero)
	}

	return
}
