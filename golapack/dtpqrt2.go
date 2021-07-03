package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtpqrt2 computes a QR factorization of a real "triangular-pentagonal"
// matrix C, which is composed of a triangular block A and pentagonal block B,
// using the compact WY representation for Q.
func Dtpqrt2(m, n, l *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, t *mat.Matrix, ldt, info *int) {
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
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *m) {
		(*info) = -7
	} else if (*ldt) < maxint(1, *n) {
		(*info) = -9
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DTPQRT2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*m) == 0 {
		return
	}

	for i = 1; i <= (*n); i++ {
		//        Generate elementary reflector H(I) to annihilate B(:,I)
		p = (*m) - (*l) + minint(*l, i)
		Dlarfg(toPtr(p+1), a.GetPtr(i-1, i-1), b.Vector(0, i-1), func() *int { y := 1; return &y }(), t.GetPtr(i-1, 0))
		if i < (*n) {
			//           W(1:N-I) := C(I:M,I+1:N)^H * C(I:M,I) [use W = T(:,N)]
			for j = 1; j <= (*n)-i; j++ {
				t.Set(j-1, (*n)-1, a.Get(i-1, i+j-1))
			}
			err = goblas.Dgemv(Trans, p, (*n)-i, one, b.Off(0, i+1-1), *ldb, b.Vector(0, i-1), 1, one, t.Vector(0, (*n)-1), 1)

			//           C(I:M,I+1:N) = C(I:m,I+1:N) + alpha*C(I:M,I)*W(1:N-1)^H
			alpha = -t.Get(i-1, 0)
			for j = 1; j <= (*n)-i; j++ {
				a.Set(i-1, i+j-1, a.Get(i-1, i+j-1)+alpha*t.Get(j-1, (*n)-1))
			}
			err = goblas.Dger(p, (*n)-i, alpha, b.Vector(0, i-1), 1, t.Vector(0, (*n)-1), 1, b.Off(0, i+1-1), *ldb)
		}
	}

	for i = 2; i <= (*n); i++ {
		//        T(1:I-1,I) := C(I:M,1:I-1)^H * (alpha * C(I:M,I))
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
		err = goblas.Dtrmv(Upper, Trans, NonUnit, p, b.Off(mp-1, 0), *ldb, t.Vector(0, i-1), 1)

		//        Rectangular part of B2
		err = goblas.Dgemv(Trans, *l, i-1-p, alpha, b.Off(mp-1, np-1), *ldb, b.Vector(mp-1, i-1), 1, zero, t.Vector(np-1, i-1), 1)

		//        B1
		err = goblas.Dgemv(Trans, (*m)-(*l), i-1, alpha, b, *ldb, b.Vector(0, i-1), 1, one, t.Vector(0, i-1), 1)

		//        T(1:I-1,I) := T(1:I-1,1:I-1) * T(1:I-1,I)
		err = goblas.Dtrmv(Upper, NoTrans, NonUnit, i-1, t, *ldt, t.Vector(0, i-1), 1)

		//        T(I,I) = tau(I)
		t.Set(i-1, i-1, t.Get(i-1, 0))
		t.Set(i-1, 0, zero)
	}
}
