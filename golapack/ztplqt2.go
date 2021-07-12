package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztplqt2 computes a LQ a factorization of a complex "triangular-pentagonal"
// matrix C, which is composed of a triangular block A and pentagonal block B,
// using the compact WY representation for Q.
func Ztplqt2(m, n, l *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, t *mat.CMatrix, ldt, info *int) {
	var alpha, one, zero complex128
	var i, j, mp, np, p int
	var err error
	_ = err

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Test the input arguments
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*l) < 0 || (*l) > min(*m, *n) {
		(*info) = -3
	} else if (*lda) < max(1, *m) {
		(*info) = -5
	} else if (*ldb) < max(1, *m) {
		(*info) = -7
	} else if (*ldt) < max(1, *m) {
		(*info) = -9
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTPLQT2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*m) == 0 {
		return
	}

	for i = 1; i <= (*m); i++ {
		//        Generate elementary reflector H(I) to annihilate B(I,:)
		p = (*n) - (*l) + min(*l, i)
		Zlarfg(toPtr(p+1), a.GetPtr(i-1, i-1), b.CVector(i-1, 0), ldb, t.GetPtr(0, i-1))
		t.Set(0, i-1, t.GetConj(0, i-1))
		if i < (*m) {
			for j = 1; j <= p; j++ {
				b.Set(i-1, j-1, b.GetConj(i-1, j-1))
			}

			//           W(M-I:1) := C(I+1:M,I:N) * C(I,I:N) [use W = T(M,:)]
			for j = 1; j <= (*m)-i; j++ {
				t.Set((*m)-1, j-1, (a.Get(i+j-1, i-1)))
			}
			err = goblas.Zgemv(NoTrans, (*m)-i, p, one, b.Off(i, 0), b.CVector(i-1, 0, *ldb), one, t.CVector((*m)-1, 0, *ldt))

			//           C(I+1:M,I:N) = C(I+1:M,I:N) + alpha * C(I,I:N)*W(M-1:1)^H
			alpha = -(t.Get(0, i-1))
			for j = 1; j <= (*m)-i; j++ {
				a.Set(i+j-1, i-1, a.Get(i+j-1, i-1)+alpha*(t.Get((*m)-1, j-1)))
			}
			err = goblas.Zgerc((*m)-i, p, alpha, t.CVector((*m)-1, 0, *ldt), b.CVector(i-1, 0, *ldb), b.Off(i, 0))
			for j = 1; j <= p; j++ {
				b.Set(i-1, j-1, b.GetConj(i-1, j-1))
			}
		}
	}

	for i = 2; i <= (*m); i++ {
		//        T(I,1:I-1) := C(I:I-1,1:N)**H * (alpha * C(I,I:N))
		alpha = -(t.Get(0, i-1))
		for j = 1; j <= i-1; j++ {
			t.Set(i-1, j-1, zero)
		}
		p = min(i-1, *l)
		np = min((*n)-(*l)+1, *n)
		mp = min(p+1, *m)
		for j = 1; j <= (*n)-(*l)+p; j++ {
			b.Set(i-1, j-1, b.GetConj(i-1, j-1))
		}

		//        Triangular part of B2
		for j = 1; j <= p; j++ {
			t.Set(i-1, j-1, (alpha * b.Get(i-1, (*n)-(*l)+j-1)))
		}
		err = goblas.Ztrmv(Lower, NoTrans, NonUnit, p, b.Off(0, np-1), t.CVector(i-1, 0, *ldt))

		//        Rectangular part of B2
		err = goblas.Zgemv(NoTrans, i-1-p, *l, alpha, b.Off(mp-1, np-1), b.CVector(i-1, np-1, *ldb), zero, t.CVector(i-1, mp-1, *ldt))

		//        B1
		err = goblas.Zgemv(NoTrans, i-1, (*n)-(*l), alpha, b, b.CVector(i-1, 0, *ldb), one, t.CVector(i-1, 0, *ldt))

		//        T(1:I-1,I) := T(1:I-1,1:I-1) * T(I,1:I-1)
		for j = 1; j <= i-1; j++ {
			t.Set(i-1, j-1, t.GetConj(i-1, j-1))
		}
		err = goblas.Ztrmv(Lower, ConjTrans, NonUnit, i-1, t, t.CVector(i-1, 0, *ldt))
		for j = 1; j <= i-1; j++ {
			t.Set(i-1, j-1, t.GetConj(i-1, j-1))
		}
		for j = 1; j <= (*n)-(*l)+p; j++ {
			b.Set(i-1, j-1, b.GetConj(i-1, j-1))
		}

		//        T(I,I) = tau(I)
		t.Set(i-1, i-1, t.Get(0, i-1))
		t.Set(0, i-1, zero)
	}
	for i = 1; i <= (*m); i++ {
		for j = i + 1; j <= (*m); j++ {
			t.Set(i-1, j-1, (t.Get(j-1, i-1)))
			t.Set(j-1, i-1, zero)
		}
	}
}
