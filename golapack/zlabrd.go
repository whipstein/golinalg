package golapack

import (
	"golinalg/goblas"
	"golinalg/mat"
)

// Zlabrd reduces the first NB rows and columns of a complex general
// m by n matrix A to upper or lower real bidiagonal form by a unitary
// transformation Q**H * A * P, and returns the matrices X and Y which
// are needed to apply the transformation to the unreduced part of A.
//
// If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower
// bidiagonal form.
//
// This is an auxiliary routine called by ZGEBRD
func Zlabrd(m, n, nb *int, a *mat.CMatrix, lda *int, d, e *mat.Vector, tauq, taup *mat.CVector, x *mat.CMatrix, ldx *int, y *mat.CMatrix, ldy *int) {
	var alpha, one, zero complex128
	var i int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Quick return if possible
	if (*m) <= 0 || (*n) <= 0 {
		return
	}

	if (*m) >= (*n) {
		//        Reduce to upper bidiagonal form
		for i = 1; i <= (*nb); i++ {
			//           Update A(i:m,i)
			Zlacgv(toPtr(i-1), y.CVector(i-1, 0), ldy)
			goblas.Zgemv(NoTrans, toPtr((*m)-i+1), toPtr(i-1), toPtrc128(-one), a.Off(i-1, 0), lda, y.CVector(i-1, 0), ldy, &one, a.CVector(i-1, i-1), func() *int { y := 1; return &y }())
			Zlacgv(toPtr(i-1), y.CVector(i-1, 0), ldy)
			goblas.Zgemv(NoTrans, toPtr((*m)-i+1), toPtr(i-1), toPtrc128(-one), x.Off(i-1, 0), ldx, a.CVector(0, i-1), func() *int { y := 1; return &y }(), &one, a.CVector(i-1, i-1), func() *int { y := 1; return &y }())

			//           Generate reflection Q(i) to annihilate A(i+1:m,i)
			alpha = a.Get(i-1, i-1)
			Zlarfg(toPtr((*m)-i+1), &alpha, a.CVector(minint(i+1, *m)-1, i-1), func() *int { y := 1; return &y }(), tauq.GetPtr(i-1))
			d.Set(i-1, real(alpha))
			if i < (*n) {
				a.Set(i-1, i-1, one)

				//              Compute Y(i+1:n,i)
				goblas.Zgemv(ConjTrans, toPtr((*m)-i+1), toPtr((*n)-i), &one, a.Off(i-1, i+1-1), lda, a.CVector(i-1, i-1), func() *int { y := 1; return &y }(), &zero, y.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(ConjTrans, toPtr((*m)-i+1), toPtr(i-1), &one, a.Off(i-1, 0), lda, a.CVector(i-1, i-1), func() *int { y := 1; return &y }(), &zero, y.CVector(0, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(NoTrans, toPtr((*n)-i), toPtr(i-1), toPtrc128(-one), y.Off(i+1-1, 0), ldy, y.CVector(0, i-1), func() *int { y := 1; return &y }(), &one, y.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(ConjTrans, toPtr((*m)-i+1), toPtr(i-1), &one, x.Off(i-1, 0), ldx, a.CVector(i-1, i-1), func() *int { y := 1; return &y }(), &zero, y.CVector(0, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(ConjTrans, toPtr(i-1), toPtr((*n)-i), toPtrc128(-one), a.Off(0, i+1-1), lda, y.CVector(0, i-1), func() *int { y := 1; return &y }(), &one, y.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				goblas.Zscal(toPtr((*n)-i), tauq.GetPtr(i-1), y.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())

				//              Update A(i,i+1:n)
				Zlacgv(toPtr((*n)-i), a.CVector(i-1, i+1-1), lda)
				Zlacgv(&i, a.CVector(i-1, 0), lda)
				goblas.Zgemv(NoTrans, toPtr((*n)-i), &i, toPtrc128(-one), y.Off(i+1-1, 0), ldy, a.CVector(i-1, 0), lda, &one, a.CVector(i-1, i+1-1), lda)
				Zlacgv(&i, a.CVector(i-1, 0), lda)
				Zlacgv(toPtr(i-1), x.CVector(i-1, 0), ldx)
				goblas.Zgemv(ConjTrans, toPtr(i-1), toPtr((*n)-i), toPtrc128(-one), a.Off(0, i+1-1), lda, x.CVector(i-1, 0), ldx, &one, a.CVector(i-1, i+1-1), lda)
				Zlacgv(toPtr(i-1), x.CVector(i-1, 0), ldx)

				//              Generate reflection P(i) to annihilate A(i,i+2:n)
				alpha = a.Get(i-1, i+1-1)
				Zlarfg(toPtr((*n)-i), &alpha, a.CVector(i-1, minint(i+2, *n)-1), lda, taup.GetPtr(i-1))
				e.Set(i-1, real(alpha))
				a.Set(i-1, i+1-1, one)

				//              Compute X(i+1:m,i)
				goblas.Zgemv(NoTrans, toPtr((*m)-i), toPtr((*n)-i), &one, a.Off(i+1-1, i+1-1), lda, a.CVector(i-1, i+1-1), lda, &zero, x.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(ConjTrans, toPtr((*n)-i), &i, &one, y.Off(i+1-1, 0), ldy, a.CVector(i-1, i+1-1), lda, &zero, x.CVector(0, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(NoTrans, toPtr((*m)-i), &i, toPtrc128(-one), a.Off(i+1-1, 0), lda, x.CVector(0, i-1), func() *int { y := 1; return &y }(), &one, x.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(NoTrans, toPtr(i-1), toPtr((*n)-i), &one, a.Off(0, i+1-1), lda, a.CVector(i-1, i+1-1), lda, &zero, x.CVector(0, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(NoTrans, toPtr((*m)-i), toPtr(i-1), toPtrc128(-one), x.Off(i+1-1, 0), ldx, x.CVector(0, i-1), func() *int { y := 1; return &y }(), &one, x.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				goblas.Zscal(toPtr((*m)-i), taup.GetPtr(i-1), x.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				Zlacgv(toPtr((*n)-i), a.CVector(i-1, i+1-1), lda)
			}
		}
	} else {
		//        Reduce to lower bidiagonal form
		for i = 1; i <= (*nb); i++ {
			//           Update A(i,i:n)
			Zlacgv(toPtr((*n)-i+1), a.CVector(i-1, i-1), lda)
			Zlacgv(toPtr(i-1), a.CVector(i-1, 0), lda)
			goblas.Zgemv(NoTrans, toPtr((*n)-i+1), toPtr(i-1), toPtrc128(-one), y.Off(i-1, 0), ldy, a.CVector(i-1, 0), lda, &one, a.CVector(i-1, i-1), lda)
			Zlacgv(toPtr(i-1), a.CVector(i-1, 0), lda)
			Zlacgv(toPtr(i-1), x.CVector(i-1, 0), ldx)
			goblas.Zgemv(ConjTrans, toPtr(i-1), toPtr((*n)-i+1), toPtrc128(-one), a.Off(0, i-1), lda, x.CVector(i-1, 0), ldx, &one, a.CVector(i-1, i-1), lda)
			Zlacgv(toPtr(i-1), x.CVector(i-1, 0), ldx)

			//           Generate reflection P(i) to annihilate A(i,i+1:n)
			alpha = a.Get(i-1, i-1)
			Zlarfg(toPtr((*n)-i+1), &alpha, a.CVector(i-1, minint(i+1, *n)-1), lda, taup.GetPtr(i-1))
			d.Set(i-1, real(alpha))
			if i < (*m) {
				a.Set(i-1, i-1, one)

				//              Compute X(i+1:m,i)
				goblas.Zgemv(NoTrans, toPtr((*m)-i), toPtr((*n)-i+1), &one, a.Off(i+1-1, i-1), lda, a.CVector(i-1, i-1), lda, &zero, x.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(ConjTrans, toPtr((*n)-i+1), toPtr(i-1), &one, y.Off(i-1, 0), ldy, a.CVector(i-1, i-1), lda, &zero, x.CVector(0, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(NoTrans, toPtr((*m)-i), toPtr(i-1), toPtrc128(-one), a.Off(i+1-1, 0), lda, x.CVector(0, i-1), func() *int { y := 1; return &y }(), &one, x.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(NoTrans, toPtr(i-1), toPtr((*n)-i+1), &one, a.Off(0, i-1), lda, a.CVector(i-1, i-1), lda, &zero, x.CVector(0, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(NoTrans, toPtr((*m)-i), toPtr(i-1), toPtrc128(-one), x.Off(i+1-1, 0), ldx, x.CVector(0, i-1), func() *int { y := 1; return &y }(), &one, x.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				goblas.Zscal(toPtr((*m)-i), taup.GetPtr(i-1), x.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				Zlacgv(toPtr((*n)-i+1), a.CVector(i-1, i-1), lda)

				//              Update A(i+1:m,i)
				Zlacgv(toPtr(i-1), y.CVector(i-1, 0), ldy)
				goblas.Zgemv(NoTrans, toPtr((*m)-i), toPtr(i-1), toPtrc128(-one), a.Off(i+1-1, 0), lda, y.CVector(i-1, 0), ldy, &one, a.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				Zlacgv(toPtr(i-1), y.CVector(i-1, 0), ldy)
				goblas.Zgemv(NoTrans, toPtr((*m)-i), &i, toPtrc128(-one), x.Off(i+1-1, 0), ldx, a.CVector(0, i-1), func() *int { y := 1; return &y }(), &one, a.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())

				//              Generate reflection Q(i) to annihilate A(i+2:m,i)
				alpha = a.Get(i+1-1, i-1)
				Zlarfg(toPtr((*m)-i), &alpha, a.CVector(minint(i+2, *m)-1, i-1), func() *int { y := 1; return &y }(), tauq.GetPtr(i-1))
				e.Set(i-1, real(alpha))
				a.Set(i+1-1, i-1, one)

				//              Compute Y(i+1:n,i)
				goblas.Zgemv(ConjTrans, toPtr((*m)-i), toPtr((*n)-i), &one, a.Off(i+1-1, i+1-1), lda, a.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), &zero, y.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(ConjTrans, toPtr((*m)-i), toPtr(i-1), &one, a.Off(i+1-1, 0), lda, a.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), &zero, y.CVector(0, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(NoTrans, toPtr((*n)-i), toPtr(i-1), toPtrc128(-one), y.Off(i+1-1, 0), ldy, y.CVector(0, i-1), func() *int { y := 1; return &y }(), &one, y.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(ConjTrans, toPtr((*m)-i), &i, &one, x.Off(i+1-1, 0), ldx, a.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), &zero, y.CVector(0, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(ConjTrans, &i, toPtr((*n)-i), toPtrc128(-one), a.Off(0, i+1-1), lda, y.CVector(0, i-1), func() *int { y := 1; return &y }(), &one, y.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				goblas.Zscal(toPtr((*n)-i), tauq.GetPtr(i-1), y.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
			} else {
				Zlacgv(toPtr((*n)-i+1), a.CVector(i-1, i-1), lda)
			}
		}
	}
}
