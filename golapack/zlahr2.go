package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Zlahr2 reduces the first NB columns of A complex general n-BY-(n-k+1)
// matrix A so that elements below the k-th subdiagonal are zero. The
// reduction is performed by an unitary similarity transformation
// Q**H * A * Q. The routine returns the matrices V and T which determine
// Q as a block reflector I - V*T*V**H, and also the matrix Y = A * V * T.
//
// This is an auxiliary routine called by ZGEHRD.
func Zlahr2(n, k, nb *int, a *mat.CMatrix, lda *int, tau *mat.CVector, t *mat.CMatrix, ldt *int, y *mat.CMatrix, ldy *int) {
	var ei, one, zero complex128
	var i int
	var err error
	_ = err

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Quick return if possible
	if (*n) <= 1 {
		return
	}

	for i = 1; i <= (*nb); i++ {
		if i > 1 {
			//           Update A(K+1:N,I)
			//
			//           Update I-th column of A - Y * V**H
			Zlacgv(toPtr(i-1), a.CVector((*k)+i-1-1, 0), lda)
			err = goblas.Zgemv(NoTrans, (*n)-(*k), i-1, -one, y.Off((*k)+1-1, 0), *ldy, a.CVector((*k)+i-1-1, 0), *lda, one, a.CVector((*k)+1-1, i-1), 1)
			Zlacgv(toPtr(i-1), a.CVector((*k)+i-1-1, 0), lda)

			//           Apply I - V * T**H * V**H to this column (call it b) from the
			//           left, using the last column of T as workspace
			//
			//           Let  V = ( V1 )   and   b = ( b1 )   (first I-1 rows)
			//                    ( V2 )             ( b2 )
			//
			//           where V1 is unit lower triangular
			//
			//           w := V1**H * b1
			goblas.Zcopy(i-1, a.CVector((*k)+1-1, i-1), 1, t.CVector(0, (*nb)-1), 1)
			err = goblas.Ztrmv(Lower, ConjTrans, Unit, i-1, a.Off((*k)+1-1, 0), *lda, t.CVector(0, (*nb)-1), 1)

			//           w := w + V2**H * b2
			err = goblas.Zgemv(ConjTrans, (*n)-(*k)-i+1, i-1, one, a.Off((*k)+i-1, 0), *lda, a.CVector((*k)+i-1, i-1), 1, one, t.CVector(0, (*nb)-1), 1)

			//           w := T**H * w
			err = goblas.Ztrmv(Upper, ConjTrans, NonUnit, i-1, t, *ldt, t.CVector(0, (*nb)-1), 1)

			//           b2 := b2 - V2*w
			err = goblas.Zgemv(NoTrans, (*n)-(*k)-i+1, i-1, -one, a.Off((*k)+i-1, 0), *lda, t.CVector(0, (*nb)-1), 1, one, a.CVector((*k)+i-1, i-1), 1)

			//           b1 := b1 - V1*w
			err = goblas.Ztrmv(Lower, NoTrans, Unit, i-1, a.Off((*k)+1-1, 0), *lda, t.CVector(0, (*nb)-1), 1)
			goblas.Zaxpy(i-1, -one, t.CVector(0, (*nb)-1), 1, a.CVector((*k)+1-1, i-1), 1)

			a.Set((*k)+i-1-1, i-1-1, ei)
		}

		//        Generate the elementary reflector H(I) to annihilate
		//        A(K+I+1:N,I)
		Zlarfg(toPtr((*n)-(*k)-i+1), a.GetPtr((*k)+i-1, i-1), a.CVector(minint((*k)+i+1, *n)-1, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1))
		ei = a.Get((*k)+i-1, i-1)
		a.Set((*k)+i-1, i-1, one)

		//        Compute  Y(K+1:N,I)
		err = goblas.Zgemv(NoTrans, (*n)-(*k), (*n)-(*k)-i+1, one, a.Off((*k)+1-1, i+1-1), *lda, a.CVector((*k)+i-1, i-1), 1, zero, y.CVector((*k)+1-1, i-1), 1)
		err = goblas.Zgemv(ConjTrans, (*n)-(*k)-i+1, i-1, one, a.Off((*k)+i-1, 0), *lda, a.CVector((*k)+i-1, i-1), 1, zero, t.CVector(0, i-1), 1)
		err = goblas.Zgemv(NoTrans, (*n)-(*k), i-1, -one, y.Off((*k)+1-1, 0), *ldy, t.CVector(0, i-1), 1, one, y.CVector((*k)+1-1, i-1), 1)
		goblas.Zscal((*n)-(*k), tau.Get(i-1), y.CVector((*k)+1-1, i-1), 1)

		//        Compute T(1:I,I)
		goblas.Zscal(i-1, -tau.Get(i-1), t.CVector(0, i-1), 1)
		err = goblas.Ztrmv(Upper, NoTrans, NonUnit, i-1, t, *ldt, t.CVector(0, i-1), 1)
		t.Set(i-1, i-1, tau.Get(i-1))

	}
	a.Set((*k)+(*nb)-1, (*nb)-1, ei)

	//     Compute Y(1:K,1:NB)
	Zlacpy('A', k, nb, a.Off(0, 1), lda, y, ldy)
	err = goblas.Ztrmm(Right, Lower, NoTrans, Unit, *k, *nb, one, a.Off((*k)+1-1, 0), *lda, y, *ldy)
	if (*n) > (*k)+(*nb) {
		err = goblas.Zgemm(NoTrans, NoTrans, *k, *nb, (*n)-(*k)-(*nb), one, a.Off(0, 2+(*nb)-1), *lda, a.Off((*k)+1+(*nb)-1, 0), *lda, one, y, *ldy)
	}
	err = goblas.Ztrmm(Right, Upper, NoTrans, NonUnit, *k, *nb, one, t, *ldt, y, *ldy)
}
