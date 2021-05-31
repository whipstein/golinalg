package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztpqrt2 computes a QR factorization of a complex "triangular-pentagonal"
// matrix C, which is composed of a triangular block A and pentagonal block B,
// using the compact WY representation for Q.
func Ztpqrt2(m, n, l *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, t *mat.CMatrix, ldt *int, info *int) {
	var alpha, one, zero complex128
	var i, j, mp, np, p int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input arguments
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*l) < 0 || (*l) > minint(*m, *n) {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *m) {
		(*info) = -7
	} else if (*ldt) < maxint(1, *n) {
		(*info) = -9
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTPQRT2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*m) == 0 {
		return
	}

	for i = 1; i <= (*n); i++ {
		//        Generate elementary reflector H(I) to annihilate B(:,I)
		p = (*m) - (*l) + minint(*l, i)
		Zlarfg(toPtr(p+1), a.GetPtr(i-1, i-1), b.CVector(0, i-1), func() *int { y := 1; return &y }(), t.GetPtr(i-1, 0))
		if i < (*n) {
			//           W(1:N-I) := C(I:M,I+1:N)**H * C(I:M,I) [use W = T(:,N)]
			for j = 1; j <= (*n)-i; j++ {
				t.Set(j-1, (*n)-1, a.GetConj(i-1, i+j-1))
			}
			goblas.Zgemv(ConjTrans, &p, toPtr((*n)-i), &one, b.Off(0, i+1-1), ldb, b.CVector(0, i-1), func() *int { y := 1; return &y }(), &one, t.CVector(0, (*n)-1), func() *int { y := 1; return &y }())

			//           C(I:M,I+1:N) = C(I:m,I+1:N) + alpha*C(I:M,I)*W(1:N-1)**H
			alpha = -t.GetConj(i-1, 0)
			for j = 1; j <= (*n)-i; j++ {
				a.Set(i-1, i+j-1, a.Get(i-1, i+j-1)+alpha*t.GetConj(j-1, (*n)-1))
			}
			goblas.Zgerc(&p, toPtr((*n)-i), &alpha, b.CVector(0, i-1), func() *int { y := 1; return &y }(), t.CVector(0, (*n)-1), func() *int { y := 1; return &y }(), b.Off(0, i+1-1), ldb)
		}
	}

	for i = 2; i <= (*n); i++ {
		//        T(1:I-1,I) := C(I:M,1:I-1)**H * (alpha * C(I:M,I))
		alpha = -t.Get(i-1, 0)
		for j = 1; j <= i-1; j++ {
			t.Set(j-1, i-1, zero)
		}
		p = minint(i-1, *l)
		mp = minint((*m)-(*l)+1, *m)
		np = minint(p+1, *n)

		//        Triangular part of B2
		for j = 1; j <= p; j++ {
			t.Set(j-1, i-1, alpha*b.Get((*m)-(*l)+j-1, i-1))
		}
		goblas.Ztrmv(Upper, ConjTrans, NonUnit, &p, b.Off(mp-1, 0), ldb, t.CVector(0, i-1), func() *int { y := 1; return &y }())

		//        Rectangular part of B2
		goblas.Zgemv(ConjTrans, l, toPtr(i-1-p), &alpha, b.Off(mp-1, np-1), ldb, b.CVector(mp-1, i-1), func() *int { y := 1; return &y }(), &zero, t.CVector(np-1, i-1), func() *int { y := 1; return &y }())

		//        B1
		goblas.Zgemv(ConjTrans, toPtr((*m)-(*l)), toPtr(i-1), &alpha, b, ldb, b.CVector(0, i-1), func() *int { y := 1; return &y }(), &one, t.CVector(0, i-1), func() *int { y := 1; return &y }())

		//        T(1:I-1,I) := T(1:I-1,1:I-1) * T(1:I-1,I)
		goblas.Ztrmv(Upper, NoTrans, NonUnit, toPtr(i-1), t, ldt, t.CVector(0, i-1), func() *int { y := 1; return &y }())

		//        T(I,I) = tau(I)
		t.Set(i-1, i-1, t.Get(i-1, 0))
		t.Set(i-1, 0, zero)
	}
}
