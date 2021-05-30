package golapack

import (
	"golinalg/goblas"
	"golinalg/mat"
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
			goblas.Zgemv(NoTrans, toPtr((*n)-(*k)), toPtr(i-1), toPtrc128(-one), y.Off((*k)+1-1, 0), ldy, a.CVector((*k)+i-1-1, 0), lda, &one, a.CVector((*k)+1-1, i-1), func() *int { y := 1; return &y }())
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
			goblas.Zcopy(toPtr(i-1), a.CVector((*k)+1-1, i-1), func() *int { y := 1; return &y }(), t.CVector(0, (*nb)-1), func() *int { y := 1; return &y }())
			goblas.Ztrmv(Lower, ConjTrans, Unit, toPtr(i-1), a.Off((*k)+1-1, 0), lda, t.CVector(0, (*nb)-1), func() *int { y := 1; return &y }())

			//           w := w + V2**H * b2
			goblas.Zgemv(ConjTrans, toPtr((*n)-(*k)-i+1), toPtr(i-1), &one, a.Off((*k)+i-1, 0), lda, a.CVector((*k)+i-1, i-1), func() *int { y := 1; return &y }(), &one, t.CVector(0, (*nb)-1), func() *int { y := 1; return &y }())

			//           w := T**H * w
			goblas.Ztrmv(Upper, ConjTrans, NonUnit, toPtr(i-1), t, ldt, t.CVector(0, (*nb)-1), func() *int { y := 1; return &y }())

			//           b2 := b2 - V2*w
			goblas.Zgemv(NoTrans, toPtr((*n)-(*k)-i+1), toPtr(i-1), toPtrc128(-one), a.Off((*k)+i-1, 0), lda, t.CVector(0, (*nb)-1), func() *int { y := 1; return &y }(), &one, a.CVector((*k)+i-1, i-1), func() *int { y := 1; return &y }())

			//           b1 := b1 - V1*w
			goblas.Ztrmv(Lower, NoTrans, Unit, toPtr(i-1), a.Off((*k)+1-1, 0), lda, t.CVector(0, (*nb)-1), func() *int { y := 1; return &y }())
			goblas.Zaxpy(toPtr(i-1), toPtrc128(-one), t.CVector(0, (*nb)-1), func() *int { y := 1; return &y }(), a.CVector((*k)+1-1, i-1), func() *int { y := 1; return &y }())

			a.Set((*k)+i-1-1, i-1-1, ei)
		}

		//        Generate the elementary reflector H(I) to annihilate
		//        A(K+I+1:N,I)
		Zlarfg(toPtr((*n)-(*k)-i+1), a.GetPtr((*k)+i-1, i-1), a.CVector(minint((*k)+i+1, *n)-1, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1))
		ei = a.Get((*k)+i-1, i-1)
		a.Set((*k)+i-1, i-1, one)

		//        Compute  Y(K+1:N,I)
		goblas.Zgemv(NoTrans, toPtr((*n)-(*k)), toPtr((*n)-(*k)-i+1), &one, a.Off((*k)+1-1, i+1-1), lda, a.CVector((*k)+i-1, i-1), func() *int { y := 1; return &y }(), &zero, y.CVector((*k)+1-1, i-1), func() *int { y := 1; return &y }())
		goblas.Zgemv(ConjTrans, toPtr((*n)-(*k)-i+1), toPtr(i-1), &one, a.Off((*k)+i-1, 0), lda, a.CVector((*k)+i-1, i-1), func() *int { y := 1; return &y }(), &zero, t.CVector(0, i-1), func() *int { y := 1; return &y }())
		goblas.Zgemv(NoTrans, toPtr((*n)-(*k)), toPtr(i-1), toPtrc128(-one), y.Off((*k)+1-1, 0), ldy, t.CVector(0, i-1), func() *int { y := 1; return &y }(), &one, y.CVector((*k)+1-1, i-1), func() *int { y := 1; return &y }())
		goblas.Zscal(toPtr((*n)-(*k)), tau.GetPtr(i-1), y.CVector((*k)+1-1, i-1), func() *int { y := 1; return &y }())

		//        Compute T(1:I,I)
		goblas.Zscal(toPtr(i-1), toPtrc128(-tau.Get(i-1)), t.CVector(0, i-1), func() *int { y := 1; return &y }())
		goblas.Ztrmv(Upper, NoTrans, NonUnit, toPtr(i-1), t, ldt, t.CVector(0, i-1), func() *int { y := 1; return &y }())
		t.Set(i-1, i-1, tau.Get(i-1))

	}
	a.Set((*k)+(*nb)-1, (*nb)-1, ei)

	//     Compute Y(1:K,1:NB)
	Zlacpy('A', k, nb, a.Off(0, 1), lda, y, ldy)
	goblas.Ztrmm(Right, Lower, NoTrans, Unit, k, nb, &one, a.Off((*k)+1-1, 0), lda, y, ldy)
	if (*n) > (*k)+(*nb) {
		goblas.Zgemm(NoTrans, NoTrans, k, nb, toPtr((*n)-(*k)-(*nb)), &one, a.Off(0, 2+(*nb)-1), lda, a.Off((*k)+1+(*nb)-1, 0), lda, &one, y, ldy)
	}
	goblas.Ztrmm(Right, Upper, NoTrans, NonUnit, k, nb, &one, t, ldt, y, ldy)
}
