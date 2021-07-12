package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dlahr2 reduces the first NB columns of A real general n-BY-(n-k+1)
// matrix A so that elements below the k-th subdiagonal are zero. The
// reduction is performed by an orthogonal similarity transformation
// Q**T * A * Q. The routine returns the matrices V and T which determine
// Q as a block reflector I - V*T*V**T, and also the matrix Y = A * V * T.
//
// This is an auxiliary routine called by DGEHRD.
func Dlahr2(n, k, nb *int, a *mat.Matrix, lda *int, tau *mat.Vector, t *mat.Matrix, ldt *int, y *mat.Matrix, ldy *int) {
	var ei, one, zero float64
	var i int
	var err error
	_ = err

	zero = 0.0
	one = 1.0

	//     Quick return if possible
	if (*n) <= 1 {
		return
	}

	for i = 1; i <= (*nb); i++ {
		if i > 1 {
			//           Update A(K+1:N,I)
			//
			//           Update I-th column of A - Y * V**T
			err = goblas.Dgemv(NoTrans, (*n)-(*k), i-1, -one, y.Off((*k), 0), a.Vector((*k)+i-1-1, 0), one, a.Vector((*k), i-1, 1))

			//           Apply I - V * T**T * V**T to this column (call it b) from the
			//           left, using the last column of T as workspace
			//
			//           Let  V = ( V1 )   and   b = ( b1 )   (first I-1 rows)
			//                    ( V2 )             ( b2 )
			//
			//           where V1 is unit lower triangular
			//
			//           w := V1**T * b1
			goblas.Dcopy(i-1, a.Vector((*k), i-1, 1), t.Vector(0, (*nb)-1, 1))
			err = goblas.Dtrmv(Lower, Trans, Unit, i-1, a.Off((*k), 0), t.Vector(0, (*nb)-1, 1))

			//           w := w + V2**T * b2
			err = goblas.Dgemv(Trans, (*n)-(*k)-i+1, i-1, one, a.Off((*k)+i-1, 0), a.Vector((*k)+i-1, i-1, 1), one, t.Vector(0, (*nb)-1, 1))

			//           w := T**T * w
			err = goblas.Dtrmv(Upper, Trans, NonUnit, i-1, t, t.Vector(0, (*nb)-1, 1))

			//           b2 := b2 - V2*w
			err = goblas.Dgemv(NoTrans, (*n)-(*k)-i+1, i-1, -one, a.Off((*k)+i-1, 0), t.Vector(0, (*nb)-1, 1), one, a.Vector((*k)+i-1, i-1, 1))

			//           b1 := b1 - V1*w
			err = goblas.Dtrmv(Lower, NoTrans, Unit, i-1, a.Off((*k), 0), t.Vector(0, (*nb)-1, 1))
			goblas.Daxpy(i-1, -one, t.Vector(0, (*nb)-1, 1), a.Vector((*k), i-1, 1))

			a.Set((*k)+i-1-1, i-1-1, ei)
		}

		//        Generate the elementary reflector H(I) to annihilate
		//        A(K+I+1:N,I)
		Dlarfg(toPtr((*n)-(*k)-i+1), a.GetPtr((*k)+i-1, i-1), a.Vector(min((*k)+i+1, *n)-1, i-1), toPtr(1), tau.GetPtr(i-1))
		ei = a.Get((*k)+i-1, i-1)
		a.Set((*k)+i-1, i-1, one)

		//        Compute  Y(K+1:N,I)
		err = goblas.Dgemv(NoTrans, (*n)-(*k), (*n)-(*k)-i+1, one, a.Off((*k), i), a.Vector((*k)+i-1, i-1, 1), zero, y.Vector((*k), i-1, 1))
		err = goblas.Dgemv(Trans, (*n)-(*k)-i+1, i-1, one, a.Off((*k)+i-1, 0), a.Vector((*k)+i-1, i-1, 1), zero, t.Vector(0, i-1, 1))
		err = goblas.Dgemv(NoTrans, (*n)-(*k), i-1, -one, y.Off((*k), 0), t.Vector(0, i-1, 1), one, y.Vector((*k), i-1, 1))
		goblas.Dscal((*n)-(*k), tau.Get(i-1), y.Vector((*k), i-1, 1))

		//        Compute T(1:I,I)
		goblas.Dscal(i-1, -tau.Get(i-1), t.Vector(0, i-1, 1))
		err = goblas.Dtrmv(Upper, NoTrans, NonUnit, i-1, t, t.Vector(0, i-1, 1))
		t.Set(i-1, i-1, tau.Get(i-1))

	}
	a.Set((*k)+(*nb)-1, (*nb)-1, ei)

	//     Compute Y(1:K,1:NB)
	Dlacpy('A', k, nb, a.Off(0, 1), lda, y, ldy)
	err = goblas.Dtrmm(Right, Lower, NoTrans, Unit, *k, *nb, one, a.Off((*k), 0), y)
	if (*n) > (*k)+(*nb) {
		err = goblas.Dgemm(NoTrans, NoTrans, *k, *nb, (*n)-(*k)-(*nb), one, a.Off(0, 2+(*nb)-1), a.Off((*k)+1+(*nb)-1, 0), one, y)
	}
	err = goblas.Dtrmm(Right, Upper, NoTrans, NonUnit, *k, *nb, one, t, y)
}
