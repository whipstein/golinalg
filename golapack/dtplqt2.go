package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtplqt2 computes a LQ a factorization of a real "triangular-pentagonal"
// matrix C, which is composed of a triangular block A and pentagonal block B,
// using the compact WY representation for Q.
func Dtplqt2(m, n, l *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, t *mat.Matrix, ldt, info *int) {
	var alpha, one, zero float64
	var i, j, mp, np, p int
	var err error
	_ = err

	one = 1.0
	zero = 0.0

	//     Test the input arguments
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*l) < 0 || (*l) > minint(*m, *n) {
		(*info) = -3
	} else if (*lda) < maxint(1, *m) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *m) {
		(*info) = -7
	} else if (*ldt) < maxint(1, *m) {
		(*info) = -9
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DTPLQT2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*m) == 0 {
		return
	}

	for i = 1; i <= (*m); i++ {
		//        Generate elementary reflector H(I) to annihilate B(I,:)
		p = (*n) - (*l) + minint(*l, i)
		Dlarfg(toPtr(p+1), a.GetPtr(i-1, i-1), b.Vector(i-1, 0), ldb, t.GetPtr(0, i-1))
		if i < (*m) {
			//           W(M-I:1) := C(I+1:M,I:N) * C(I,I:N) [use W = T(M,:)]
			for j = 1; j <= (*m)-i; j++ {
				t.Set((*m)-1, j-1, a.Get(i+j-1, i-1))
			}
			err = goblas.Dgemv(NoTrans, (*m)-i, p, one, b.Off(i+1-1, 0), *ldb, b.Vector(i-1, 0), *ldb, one, t.Vector((*m)-1, 0), *ldt)

			//           C(I+1:M,I:N) = C(I+1:M,I:N) + alpha * C(I,I:N)*W(M-1:1)^H
			alpha = -t.Get(0, i-1)
			for j = 1; j <= (*m)-i; j++ {
				a.Set(i+j-1, i-1, a.Get(i+j-1, i-1)+alpha*t.Get((*m)-1, j-1))
			}
			err = goblas.Dger((*m)-i, p, alpha, t.Vector((*m)-1, 0), *ldt, b.Vector(i-1, 0), *ldb, b.Off(i+1-1, 0), *ldb)
		}
	}

	for i = 2; i <= (*m); i++ {
		//        T(I,1:I-1) := C(I:I-1,1:N) * (alpha * C(I,I:N)^H)
		alpha = -t.Get(0, i-1)
		for j = 1; j <= i-1; j++ {
			t.Set(i-1, j-1, zero)
		}
		p = minint(i-1, *l)
		np = minint((*n)-(*l)+1, *n)
		mp = minint(p+1, *m)

		//        Triangular part of B2
		for j = 1; j <= p; j++ {
			t.Set(i-1, j-1, alpha*b.Get(i-1, (*n)-(*l)+j-1))
		}
		err = goblas.Dtrmv(Lower, NoTrans, NonUnit, p, b.Off(0, np-1), *ldb, t.Vector(i-1, 0), *ldt)

		//        Rectangular part of B2
		err = goblas.Dgemv(NoTrans, i-1-p, *l, alpha, b.Off(mp-1, np-1), *ldb, b.Vector(i-1, np-1), *ldb, zero, t.Vector(i-1, mp-1), *ldt)

		//        B1
		err = goblas.Dgemv(NoTrans, i-1, (*n)-(*l), alpha, b, *ldb, b.Vector(i-1, 0), *ldb, one, t.Vector(i-1, 0), *ldt)

		//        T(1:I-1,I) := T(1:I-1,1:I-1) * T(I,1:I-1)
		err = goblas.Dtrmv(Lower, Trans, NonUnit, i-1, t, *ldt, t.Vector(i-1, 0), *ldt)

		//        T(I,I) = tau(I)
		t.Set(i-1, i-1, t.Get(0, i-1))
		t.Set(0, i-1, zero)
	}
	for i = 1; i <= (*m); i++ {
		for j = i + 1; j <= (*m); j++ {
			t.Set(i-1, j-1, t.Get(j-1, i-1))
			t.Set(j-1, i-1, zero)
		}
	}
}
