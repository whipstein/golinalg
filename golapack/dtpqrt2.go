package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtpqrt2 computes a QR factorization of a real "triangular-pentagonal"
// matrix C, which is composed of a triangular block A and pentagonal block B,
// using the compact WY representation for Q.
func Dtpqrt2(m, n, l int, a, b, t *mat.Matrix) (err error) {
	var alpha, one, zero float64
	var i, j, mp, np, p int

	one = 1.0
	zero = 0.0

	//     Test the input arguments
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if l < 0 || l > min(m, n) {
		err = fmt.Errorf("l < 0 || l > min(m, n): l=%v, m=%v, n=%v", l, m, n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, m) {
		err = fmt.Errorf("b.Rows < max(1, m): b.Rows=%v, m=%v", b.Rows, m)
	} else if t.Rows < max(1, n) {
		err = fmt.Errorf("t.Rows < max(1, n): t.Rows=%v, n=%v", t.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dtpqrt2", err)
		return
	}

	//     Quick return if possible
	if n == 0 || m == 0 {
		return
	}

	for i = 1; i <= n; i++ {
		//        Generate elementary reflector H(I) to annihilate B(:,I)
		p = m - l + min(l, i)
		*a.GetPtr(i-1, i-1), *t.GetPtr(i-1, 0) = Dlarfg(p+1, a.Get(i-1, i-1), b.Vector(0, i-1, 1))
		if i < n {
			//           W(1:N-I) := C(I:M,I+1:N)^H * C(I:M,I) [use W = T(:,N)]
			for j = 1; j <= n-i; j++ {
				t.Set(j-1, n-1, a.Get(i-1, i+j-1))
			}
			if err = goblas.Dgemv(Trans, p, n-i, one, b.Off(0, i), b.Vector(0, i-1, 1), one, t.Vector(0, n-1, 1)); err != nil {
				panic(err)
			}

			//           C(I:M,I+1:N) = C(I:m,I+1:N) + alpha*C(I:M,I)*W(1:N-1)^H
			alpha = -t.Get(i-1, 0)
			for j = 1; j <= n-i; j++ {
				a.Set(i-1, i+j-1, a.Get(i-1, i+j-1)+alpha*t.Get(j-1, n-1))
			}
			if err = goblas.Dger(p, n-i, alpha, b.Vector(0, i-1, 1), t.Vector(0, n-1, 1), b.Off(0, i)); err != nil {
				panic(err)
			}
		}
	}

	for i = 2; i <= n; i++ {
		//        T(1:I-1,I) := C(I:M,1:I-1)^H * (alpha * C(I:M,I))
		alpha = -t.Get(i-1, 0)
		for j = 1; j <= i-1; j++ {
			t.Set(j-1, i-1, zero)
		}
		p = min(i-1, l)
		mp = min(m-l+1, m)
		np = min(p+1, n)

		//        Triangular part of B2
		for j = 1; j <= p; j++ {
			t.Set(j-1, i-1, alpha*b.Get(m-l+j-1, i-1))
		}
		if err = goblas.Dtrmv(Upper, Trans, NonUnit, p, b.Off(mp-1, 0), t.Vector(0, i-1, 1)); err != nil {
			panic(err)
		}

		//        Rectangular part of B2
		if err = goblas.Dgemv(Trans, l, i-1-p, alpha, b.Off(mp-1, np-1), b.Vector(mp-1, i-1, 1), zero, t.Vector(np-1, i-1, 1)); err != nil {
			panic(err)
		}

		//        B1
		if err = goblas.Dgemv(Trans, m-l, i-1, alpha, b, b.Vector(0, i-1, 1), one, t.Vector(0, i-1, 1)); err != nil {
			panic(err)
		}

		//        T(1:I-1,I) := T(1:I-1,1:I-1) * T(1:I-1,I)
		if err = goblas.Dtrmv(Upper, NoTrans, NonUnit, i-1, t, t.Vector(0, i-1, 1)); err != nil {
			panic(err)
		}

		//        T(I,I) = tau(I)
		t.Set(i-1, i-1, t.Get(i-1, 0))
		t.Set(i-1, 0, zero)
	}

	return
}
