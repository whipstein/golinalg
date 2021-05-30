package golapack

import (
	"golinalg/goblas"
	"golinalg/mat"
)

// Zlatrd reduces NB rows and columns of a complex Hermitian matrix A to
// Hermitian tridiagonal form by a unitary similarity
// transformation Q**H * A * Q, and returns the matrices V and W which are
// needed to apply the transformation to the unreduced part of A.
//
// If UPLO = 'U', ZLATRD reduces the last NB rows and columns of a
// matrix, of which the upper triangle is supplied;
// if UPLO = 'L', ZLATRD reduces the first NB rows and columns of a
// matrix, of which the lower triangle is supplied.
//
// This is an auxiliary routine called by ZHETRD.
func Zlatrd(uplo byte, n, nb *int, a *mat.CMatrix, lda *int, e *mat.Vector, tau *mat.CVector, w *mat.CMatrix, ldw *int) {
	var alpha, half, one, zero complex128
	var i, iw int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	half = (0.5 + 0.0*1i)

	//     Quick return if possible
	if (*n) <= 0 {
		return
	}

	if uplo == 'U' {
		//        Reduce last NB columns of upper triangle
		for i = (*n); i >= (*n)-(*nb)+1; i-- {
			iw = i - (*n) + (*nb)
			if i < (*n) {
				//              Update A(1:i,i)
				a.Set(i-1, i-1, a.GetReCmplx(i-1, i-1))
				Zlacgv(toPtr((*n)-i), w.CVector(i-1, iw+1-1), ldw)
				goblas.Zgemv(NoTrans, &i, toPtr((*n)-i), toPtrc128(-one), a.Off(0, i+1-1), lda, w.CVector(i-1, iw+1-1), ldw, &one, a.CVector(0, i-1), func() *int { y := 1; return &y }())
				Zlacgv(toPtr((*n)-i), w.CVector(i-1, iw+1-1), ldw)
				Zlacgv(toPtr((*n)-i), a.CVector(i-1, i+1-1), lda)
				goblas.Zgemv(NoTrans, &i, toPtr((*n)-i), toPtrc128(-one), w.Off(0, iw+1-1), ldw, a.CVector(i-1, i+1-1), lda, &one, a.CVector(0, i-1), func() *int { y := 1; return &y }())
				Zlacgv(toPtr((*n)-i), a.CVector(i-1, i+1-1), lda)
				a.Set(i-1, i-1, a.GetReCmplx(i-1, i-1))
			}
			if i > 1 {
				//              Generate elementary reflector H(i) to annihilate
				//              A(1:i-2,i)
				alpha = a.Get(i-1-1, i-1)
				Zlarfg(toPtr(i-1), &alpha, a.CVector(0, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1-1))
				e.Set(i-1-1, real(alpha))
				a.Set(i-1-1, i-1, one)

				//              Compute W(1:i-1,i)
				goblas.Zhemv(Upper, toPtr(i-1), &one, a, lda, a.CVector(0, i-1), func() *int { y := 1; return &y }(), &zero, w.CVector(0, iw-1), func() *int { y := 1; return &y }())
				if i < (*n) {
					goblas.Zgemv(ConjTrans, toPtr(i-1), toPtr((*n)-i), &one, w.Off(0, iw+1-1), ldw, a.CVector(0, i-1), func() *int { y := 1; return &y }(), &zero, w.CVector(i+1-1, iw-1), func() *int { y := 1; return &y }())
					goblas.Zgemv(NoTrans, toPtr(i-1), toPtr((*n)-i), toPtrc128(-one), a.Off(0, i+1-1), lda, w.CVector(i+1-1, iw-1), func() *int { y := 1; return &y }(), &one, w.CVector(0, iw-1), func() *int { y := 1; return &y }())
					goblas.Zgemv(ConjTrans, toPtr(i-1), toPtr((*n)-i), &one, a.Off(0, i+1-1), lda, a.CVector(0, i-1), func() *int { y := 1; return &y }(), &zero, w.CVector(i+1-1, iw-1), func() *int { y := 1; return &y }())
					goblas.Zgemv(NoTrans, toPtr(i-1), toPtr((*n)-i), toPtrc128(-one), w.Off(0, iw+1-1), ldw, w.CVector(i+1-1, iw-1), func() *int { y := 1; return &y }(), &one, w.CVector(0, iw-1), func() *int { y := 1; return &y }())
				}
				goblas.Zscal(toPtr(i-1), tau.GetPtr(i-1-1), w.CVector(0, iw-1), func() *int { y := 1; return &y }())
				alpha = -half * tau.Get(i-1-1) * goblas.Zdotc(toPtr(i-1), w.CVector(0, iw-1), func() *int { y := 1; return &y }(), a.CVector(0, i-1), func() *int { y := 1; return &y }())
				goblas.Zaxpy(toPtr(i-1), &alpha, a.CVector(0, i-1), func() *int { y := 1; return &y }(), w.CVector(0, iw-1), func() *int { y := 1; return &y }())
			}

		}
	} else {
		//        Reduce first NB columns of lower triangle
		for i = 1; i <= (*nb); i++ {
			//           Update A(i:n,i)
			a.Set(i-1, i-1, a.GetReCmplx(i-1, i-1))
			Zlacgv(toPtr(i-1), w.CVector(i-1, 0), ldw)
			goblas.Zgemv(NoTrans, toPtr((*n)-i+1), toPtr(i-1), toPtrc128(-one), a.Off(i-1, 0), lda, w.CVector(i-1, 0), ldw, &one, a.CVector(i-1, i-1), func() *int { y := 1; return &y }())
			Zlacgv(toPtr(i-1), w.CVector(i-1, 0), ldw)
			Zlacgv(toPtr(i-1), a.CVector(i-1, 0), lda)
			goblas.Zgemv(NoTrans, toPtr((*n)-i+1), toPtr(i-1), toPtrc128(-one), w.Off(i-1, 0), ldw, a.CVector(i-1, 0), lda, &one, a.CVector(i-1, i-1), func() *int { y := 1; return &y }())
			Zlacgv(toPtr(i-1), a.CVector(i-1, 0), lda)
			a.Set(i-1, i-1, a.GetReCmplx(i-1, i-1))
			if i < (*n) {

				//              Generate elementary reflector H(i) to annihilate
				//              A(i+2:n,i)
				alpha = a.Get(i+1-1, i-1)
				Zlarfg(toPtr((*n)-i), &alpha, a.CVector(minint(i+2, *n)-1, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1))
				e.Set(i-1, real(alpha))
				a.Set(i+1-1, i-1, one)

				//              Compute W(i+1:n,i)
				goblas.Zhemv(Lower, toPtr((*n)-i), &one, a.Off(i+1-1, i+1-1), lda, a.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), &zero, w.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(ConjTrans, toPtr((*n)-i), toPtr(i-1), &one, w.Off(i+1-1, 0), ldw, a.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), &zero, w.CVector(0, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(NoTrans, toPtr((*n)-i), toPtr(i-1), toPtrc128(-one), a.Off(i+1-1, 0), lda, w.CVector(0, i-1), func() *int { y := 1; return &y }(), &one, w.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(ConjTrans, toPtr((*n)-i), toPtr(i-1), &one, a.Off(i+1-1, 0), lda, a.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), &zero, w.CVector(0, i-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(NoTrans, toPtr((*n)-i), toPtr(i-1), toPtrc128(-one), w.Off(i+1-1, 0), ldw, w.CVector(0, i-1), func() *int { y := 1; return &y }(), &one, w.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				goblas.Zscal(toPtr((*n)-i), tau.GetPtr(i-1), w.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				alpha = -half * tau.Get(i-1) * goblas.Zdotc(toPtr((*n)-i), w.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), a.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				goblas.Zaxpy(toPtr((*n)-i), &alpha, a.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), w.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
			}

		}
	}
}
