package golapack

import (
	"golinalg/goblas"
	"golinalg/mat"
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
			goblas.Dgemv(NoTrans, toPtr((*n)-(*k)), toPtr(i-1), toPtrf64(-one), y.Off((*k)+1-1, 0), ldy, a.Vector((*k)+i-1-1, 0), lda, &one, a.Vector((*k)+1-1, i-1), toPtr(1))

			//           Apply I - V * T**T * V**T to this column (call it b) from the
			//           left, using the last column of T as workspace
			//
			//           Let  V = ( V1 )   and   b = ( b1 )   (first I-1 rows)
			//                    ( V2 )             ( b2 )
			//
			//           where V1 is unit lower triangular
			//
			//           w := V1**T * b1
			goblas.Dcopy(toPtr(i-1), a.Vector((*k)+1-1, i-1), toPtr(1), t.Vector(0, (*nb)-1), toPtr(1))
			goblas.Dtrmv(Lower, Trans, Unit, toPtr(i-1), a.Off((*k)+1-1, 0), lda, t.Vector(0, (*nb)-1), toPtr(1))

			//           w := w + V2**T * b2
			goblas.Dgemv(Trans, toPtr((*n)-(*k)-i+1), toPtr(i-1), &one, a.Off((*k)+i-1, 0), lda, a.Vector((*k)+i-1, i-1), toPtr(1), &one, t.Vector(0, (*nb)-1), toPtr(1))

			//           w := T**T * w
			goblas.Dtrmv(Upper, Trans, NonUnit, toPtr(i-1), t, ldt, t.Vector(0, (*nb)-1), toPtr(1))

			//           b2 := b2 - V2*w
			goblas.Dgemv(NoTrans, toPtr((*n)-(*k)-i+1), toPtr(i-1), toPtrf64(-one), a.Off((*k)+i-1, 0), lda, t.Vector(0, (*nb)-1), toPtr(1), &one, a.Vector((*k)+i-1, i-1), toPtr(1))

			//           b1 := b1 - V1*w
			goblas.Dtrmv(Lower, NoTrans, Unit, toPtr(i-1), a.Off((*k)+1-1, 0), lda, t.Vector(0, (*nb)-1), toPtr(1))
			goblas.Daxpy(toPtr(i-1), toPtrf64(-one), t.Vector(0, (*nb)-1), toPtr(1), a.Vector((*k)+1-1, i-1), toPtr(1))

			a.Set((*k)+i-1-1, i-1-1, ei)
		}

		//        Generate the elementary reflector H(I) to annihilate
		//        A(K+I+1:N,I)
		Dlarfg(toPtr((*n)-(*k)-i+1), a.GetPtr((*k)+i-1, i-1), a.Vector(minint((*k)+i+1, *n)-1, i-1), toPtr(1), tau.GetPtr(i-1))
		ei = a.Get((*k)+i-1, i-1)
		a.Set((*k)+i-1, i-1, one)

		//        Compute  Y(K+1:N,I)
		goblas.Dgemv(NoTrans, toPtr((*n)-(*k)), toPtr((*n)-(*k)-i+1), &one, a.Off((*k)+1-1, i+1-1), lda, a.Vector((*k)+i-1, i-1), toPtr(1), &zero, y.Vector((*k)+1-1, i-1), toPtr(1))
		goblas.Dgemv(Trans, toPtr((*n)-(*k)-i+1), toPtr(i-1), &one, a.Off((*k)+i-1, 0), lda, a.Vector((*k)+i-1, i-1), toPtr(1), &zero, t.Vector(0, i-1), toPtr(1))
		goblas.Dgemv(NoTrans, toPtr((*n)-(*k)), toPtr(i-1), toPtrf64(-one), y.Off((*k)+1-1, 0), ldy, t.Vector(0, i-1), toPtr(1), &one, y.Vector((*k)+1-1, i-1), toPtr(1))
		goblas.Dscal(toPtr((*n)-(*k)), tau.GetPtr(i-1), y.Vector((*k)+1-1, i-1), toPtr(1))

		//        Compute T(1:I,I)
		goblas.Dscal(toPtr(i-1), toPtrf64(-tau.Get(i-1)), t.Vector(0, i-1), toPtr(1))
		goblas.Dtrmv(Upper, NoTrans, NonUnit, toPtr(i-1), t, ldt, t.Vector(0, i-1), toPtr(1))
		t.Set(i-1, i-1, tau.Get(i-1))

	}
	a.Set((*k)+(*nb)-1, (*nb)-1, ei)

	//     Compute Y(1:K,1:NB)
	Dlacpy('A', k, nb, a.Off(0, 1), lda, y, ldy)
	goblas.Dtrmm(Right, Lower, NoTrans, Unit, k, nb, &one, a.Off((*k)+1-1, 0), lda, y, ldy)
	if (*n) > (*k)+(*nb) {
		goblas.Dgemm(NoTrans, NoTrans, k, nb, toPtr((*n)-(*k)-(*nb)), &one, a.Off(0, 2+(*nb)-1), lda, a.Off((*k)+1+(*nb)-1, 0), lda, &one, y, ldy)
	}
	goblas.Dtrmm(Right, Upper, NoTrans, NonUnit, k, nb, &one, t, ldt, y, ldy)
}
