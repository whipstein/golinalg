package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dgeqrt2 computes a QR factorization of a real M-by-N matrix A,
// using the compact WY representation of Q.
func Dgeqrt2(m, n *int, a *mat.Matrix, lda *int, t *mat.Matrix, ldt, info *int) {
	var aii, alpha, one, zero float64
	var i, k int

	one = 1.0e+00
	zero = 0.0e+00

	//     Test the input arguments
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *m) {
		(*info) = -4
	} else if (*ldt) < maxint(1, *n) {
		(*info) = -6
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGEQRT2"), -(*info))
		return
	}

	k = minint(*m, *n)

	for i = 1; i <= k; i++ {
		//        Generate elem. refl. H(i) to annihilate A(i+1:m,i), tau(I) -> T(I,1)
		Dlarfg(toPtr((*m)-i+1), a.GetPtr(i-1, i-1), a.Vector(minint(i+1, *m)-1, i-1), func() *int { y := 1; return &y }(), t.GetPtr(i-1, 0))
		if i < (*n) {
			//           Apply H(i) to A(I:M,I+1:N) from the left
			aii = a.Get(i-1, i-1)
			a.Set(i-1, i-1, one)

			//           W(1:N-I) := A(I:M,I+1:N)^H * A(I:M,I) [W = T(:,N)]
			goblas.Dgemv(Trans, toPtr((*m)-i+1), toPtr((*n)-i), &one, a.Off(i-1, i+1-1), lda, a.Vector(i-1, i-1), toPtr(1), &zero, t.Vector(0, (*n)-1), toPtr(1))

			//           A(I:M,I+1:N) = A(I:m,I+1:N) + alpha*A(I:M,I)*W(1:N-1)^H
			alpha = -t.Get(i-1, 0)
			goblas.Dger(toPtr((*m)-i+1), toPtr((*n)-i), &alpha, a.Vector(i-1, i-1), toPtr(1), t.Vector(0, (*n)-1), toPtr(1), a.Off(i-1, i+1-1), lda)
			a.Set(i-1, i-1, aii)
		}
	}

	for i = 2; i <= (*n); i++ {
		aii = a.Get(i-1, i-1)
		a.Set(i-1, i-1, one)

		//        T(1:I-1,I) := alpha * A(I:M,1:I-1)**T * A(I:M,I)
		alpha = -t.Get(i-1, 0)
		goblas.Dgemv(Trans, toPtr((*m)-i+1), toPtr(i-1), &alpha, a.Off(i-1, 0), lda, a.Vector(i-1, i-1), toPtr(1), &zero, t.Vector(0, i-1), toPtr(1))
		a.Set(i-1, i-1, aii)

		//        T(1:I-1,I) := T(1:I-1,1:I-1) * T(1:I-1,I)
		goblas.Dtrmv(Upper, NoTrans, NonUnit, toPtr(i-1), t, ldt, t.Vector(0, i-1), toPtr(1))

		//           T(I,I) = tau(I)
		t.Set(i-1, i-1, t.Get(i-1, 0))
		t.Set(i-1, 0, zero)
	}
}
