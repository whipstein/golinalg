package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgeqrt2 computes a QR factorization of a complex M-by-N matrix A,
// using the compact WY representation of Q.
func Zgeqrt2(m, n *int, a *mat.CMatrix, lda *int, t *mat.CMatrix, ldt, info *int) {
	var aii, alpha, one, zero complex128
	var i, k int
	var err error
	_ = err

	one = (1.0e+00 + 0.0e+00*1i)
	zero = (0.0e+00 + 0.0e+00*1i)

	//     Test the input arguments
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *m) {
		(*info) = -4
	} else if (*ldt) < max(1, *n) {
		(*info) = -6
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGEQRT2"), -(*info))
		return
	}

	k = min(*m, *n)

	for i = 1; i <= k; i++ {
		//        Generate elem. refl. H(i) to annihilate A(i+1:m,i), tau(I) -> T(I,1)
		Zlarfg(toPtr((*m)-i+1), a.GetPtr(i-1, i-1), a.CVector(min(i+1, *m)-1, i-1), func() *int { y := 1; return &y }(), t.GetPtr(i-1, 0))
		if i < (*n) {
			//           Apply H(i) to A(I:M,I+1:N) from the left
			aii = a.Get(i-1, i-1)
			a.Set(i-1, i-1, one)

			//           W(1:N-I) := A(I:M,I+1:N)^H * A(I:M,I) [W = T(:,N)]
			err = goblas.Zgemv(ConjTrans, (*m)-i+1, (*n)-i, one, a.Off(i-1, i), a.CVector(i-1, i-1, 1), zero, t.CVector(0, (*n)-1, 1))

			//           A(I:M,I+1:N) = A(I:m,I+1:N) + alpha*A(I:M,I)*W(1:N-1)^H
			alpha = -t.GetConj(i-1, 0)
			err = goblas.Zgerc((*m)-i+1, (*n)-i, alpha, a.CVector(i-1, i-1, 1), t.CVector(0, (*n)-1, 1), a.Off(i-1, i))
			a.Set(i-1, i-1, aii)
		}
	}

	for i = 2; i <= (*n); i++ {
		aii = a.Get(i-1, i-1)
		a.Set(i-1, i-1, one)

		//        T(1:I-1,I) := alpha * A(I:M,1:I-1)**H * A(I:M,I)
		alpha = -t.Get(i-1, 0)
		err = goblas.Zgemv(ConjTrans, (*m)-i+1, i-1, alpha, a.Off(i-1, 0), a.CVector(i-1, i-1, 1), zero, t.CVector(0, i-1, 1))
		a.Set(i-1, i-1, aii)

		//        T(1:I-1,I) := T(1:I-1,1:I-1) * T(1:I-1,I)
		err = goblas.Ztrmv(Upper, NoTrans, NonUnit, i-1, t, t.CVector(0, i-1, 1))

		//           T(I,I) = tau(I)
		t.Set(i-1, i-1, t.Get(i-1, 0))
		t.Set(i-1, 0, zero)
	}
}
