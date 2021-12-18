package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztplqt2 computes a LQ a factorization of a complex "triangular-pentagonal"
// matrix C, which is composed of a triangular block A and pentagonal block B,
// using the compact WY representation for Q.
func Ztplqt2(m, n, l int, a, b, t *mat.CMatrix) (err error) {
	var alpha, one, zero complex128
	var i, j, mp, np, p int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Test the input arguments
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if l < 0 || l > min(m, n) {
		err = fmt.Errorf("l < 0 || l > min(m, n): l=%v, m=%v, n=%v", l, m, n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if b.Rows < max(1, m) {
		err = fmt.Errorf("b.Rows < max(1, m): b.Rows=%v, m=%v", b.Rows, m)
	} else if t.Rows < max(1, m) {
		err = fmt.Errorf("t.Rows < max(1, m): t.Rows=%v, m=%v", t.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("Ztplqt2", err)
		return
	}

	//     Quick return if possible
	if n == 0 || m == 0 {
		return
	}

	for i = 1; i <= m; i++ {
		//        Generate elementary reflector H(I) to annihilate B(I,:)
		p = n - l + min(l, i)
		*a.GetPtr(i-1, i-1), *t.GetPtr(0, i-1) = Zlarfg(p+1, a.Get(i-1, i-1), b.Off(i-1, 0).CVector(), b.Rows)
		t.Set(0, i-1, t.GetConj(0, i-1))
		if i < m {
			for j = 1; j <= p; j++ {
				b.Set(i-1, j-1, b.GetConj(i-1, j-1))
			}

			//           W(M-I:1) := C(I+1:M,I:N) * C(I,I:N) [use W = T(M,:)]
			for j = 1; j <= m-i; j++ {
				t.Set(m-1, j-1, (a.Get(i+j-1, i-1)))
			}
			err = t.Off(m-1, 0).CVector().Gemv(NoTrans, m-i, p, one, b.Off(i, 0), b.Off(i-1, 0).CVector(), b.Rows, one, t.Rows)

			//           C(I+1:M,I:N) = C(I+1:M,I:N) + alpha * C(I,I:N)*W(M-1:1)^H
			alpha = -(t.Get(0, i-1))
			for j = 1; j <= m-i; j++ {
				a.Set(i+j-1, i-1, a.Get(i+j-1, i-1)+alpha*(t.Get(m-1, j-1)))
			}
			err = b.Off(i, 0).Gerc(m-i, p, alpha, t.Off(m-1, 0).CVector(), t.Rows, b.Off(i-1, 0).CVector(), b.Rows)
			for j = 1; j <= p; j++ {
				b.Set(i-1, j-1, b.GetConj(i-1, j-1))
			}
		}
	}

	for i = 2; i <= m; i++ {
		//        T(I,1:I-1) := C(I:I-1,1:N)**H * (alpha * C(I,I:N))
		alpha = -(t.Get(0, i-1))
		for j = 1; j <= i-1; j++ {
			t.Set(i-1, j-1, zero)
		}
		p = min(i-1, l)
		np = min(n-l+1, n)
		mp = min(p+1, m)
		for j = 1; j <= n-l+p; j++ {
			b.Set(i-1, j-1, b.GetConj(i-1, j-1))
		}

		//        Triangular part of B2
		for j = 1; j <= p; j++ {
			t.Set(i-1, j-1, (alpha * b.Get(i-1, n-l+j-1)))
		}
		err = t.Off(i-1, 0).CVector().Trmv(Lower, NoTrans, NonUnit, p, b.Off(0, np-1), t.Rows)

		//        Rectangular part of B2
		err = t.Off(i-1, mp-1).CVector().Gemv(NoTrans, i-1-p, l, alpha, b.Off(mp-1, np-1), b.Off(i-1, np-1).CVector(), b.Rows, zero, t.Rows)

		//        B1
		err = t.Off(i-1, 0).CVector().Gemv(NoTrans, i-1, n-l, alpha, b, b.Off(i-1, 0).CVector(), b.Rows, one, t.Rows)

		//        T(1:I-1,I) := T(1:I-1,1:I-1) * T(I,1:I-1)
		for j = 1; j <= i-1; j++ {
			t.Set(i-1, j-1, t.GetConj(i-1, j-1))
		}
		err = t.Off(i-1, 0).CVector().Trmv(Lower, ConjTrans, NonUnit, i-1, t, t.Rows)
		for j = 1; j <= i-1; j++ {
			t.Set(i-1, j-1, t.GetConj(i-1, j-1))
		}
		for j = 1; j <= n-l+p; j++ {
			b.Set(i-1, j-1, b.GetConj(i-1, j-1))
		}

		//        T(I,I) = tau(I)
		t.Set(i-1, i-1, t.Get(0, i-1))
		t.Set(0, i-1, zero)
	}
	for i = 1; i <= m; i++ {
		for j = i + 1; j <= m; j++ {
			t.Set(i-1, j-1, (t.Get(j-1, i-1)))
			t.Set(j-1, i-1, zero)
		}
	}

	return
}
