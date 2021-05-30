package golapack

import (
	"golinalg/goblas"
	"golinalg/mat"
)

// Dlatrd reduces NB rows and columns of a real symmetric matrix A to
// symmetric tridiagonal form by an orthogonal similarity
// transformation Q**T * A * Q, and returns the matrices V and W which are
// needed to apply the transformation to the unreduced part of A.
//
// If UPLO = 'U', DLATRD reduces the last NB rows and columns of a
// matrix, of which the upper triangle is supplied;
// if UPLO = 'L', DLATRD reduces the first NB rows and columns of a
// matrix, of which the lower triangle is supplied.
//
// This is an auxiliary routine called by DSYTRD.
func Dlatrd(uplo byte, n, nb *int, a *mat.Matrix, lda *int, e, tau *mat.Vector, w *mat.Matrix, ldw *int) {
	var alpha, half, one, zero float64
	var i, iw int

	zero = 0.0
	one = 1.0
	half = 0.5

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
				goblas.Dgemv(NoTrans, &i, toPtr((*n)-i), toPtrf64(-one), a.Off(0, i+1-1), lda, w.Vector(i-1, iw+1-1), ldw, &one, a.Vector(0, i-1), toPtr(1))
				goblas.Dgemv(NoTrans, &i, toPtr((*n)-i), toPtrf64(-one), w.Off(0, iw+1-1), ldw, a.Vector(i-1, i+1-1), lda, &one, a.Vector(0, i-1), toPtr(1))
			}
			if i > 1 {
				//              Generate elementary reflector H(i) to annihilate
				//              A(1:i-2,i)
				Dlarfg(toPtr(i-1), a.GetPtr(i-1-1, i-1), a.Vector(0, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1-1))
				e.Set(i-1-1, a.Get(i-1-1, i-1))
				a.Set(i-1-1, i-1, one)

				//              Compute W(1:i-1,i)
				goblas.Dsymv(Upper, toPtr(i-1), &one, a, lda, a.Vector(0, i-1), toPtr(1), &zero, w.Vector(0, iw-1), toPtr(1))
				if i < (*n) {
					goblas.Dgemv(Trans, toPtr(i-1), toPtr((*n)-i), &one, w.Off(0, iw+1-1), ldw, a.Vector(0, i-1), toPtr(1), &zero, w.Vector(i+1-1, iw-1), toPtr(1))
					goblas.Dgemv(NoTrans, toPtr(i-1), toPtr((*n)-i), toPtrf64(-one), a.Off(0, i+1-1), lda, w.Vector(i+1-1, iw-1), toPtr(1), &one, w.Vector(0, iw-1), toPtr(1))
					goblas.Dgemv(Trans, toPtr(i-1), toPtr((*n)-i), &one, a.Off(0, i+1-1), lda, a.Vector(0, i-1), toPtr(1), &zero, w.Vector(i+1-1, iw-1), toPtr(1))
					goblas.Dgemv(NoTrans, toPtr(i-1), toPtr((*n)-i), toPtrf64(-one), w.Off(0, iw+1-1), ldw, w.Vector(i+1-1, iw-1), toPtr(1), &one, w.Vector(0, iw-1), toPtr(1))
				}
				goblas.Dscal(toPtr(i-1), tau.GetPtr(i-1-1), w.Vector(0, iw-1), toPtr(1))
				alpha = -half * tau.Get(i-1-1) * goblas.Ddot(toPtr(i-1), w.Vector(0, iw-1), toPtr(1), a.Vector(0, i-1), toPtr(1))
				goblas.Daxpy(toPtr(i-1), &alpha, a.Vector(0, i-1), toPtr(1), w.Vector(0, iw-1), toPtr(1))
			}

		}
	} else {
		//        Reduce first NB columns of lower triangle
		for i = 1; i <= (*nb); i++ {
			//           Update A(i:n,i)
			goblas.Dgemv(NoTrans, toPtr((*n)-i+1), toPtr(i-1), toPtrf64(-one), a.Off(i-1, 0), lda, w.Vector(i-1, 0), ldw, &one, a.Vector(i-1, i-1), toPtr(1))
			goblas.Dgemv(NoTrans, toPtr((*n)-i+1), toPtr(i-1), toPtrf64(-one), w.Off(i-1, 0), ldw, a.Vector(i-1, 0), lda, &one, a.Vector(i-1, i-1), toPtr(1))
			if i < (*n) {
				//              Generate elementary reflector H(i) to annihilate
				//              A(i+2:n,i)
				Dlarfg(toPtr((*n)-i), a.GetPtr(i+1-1, i-1), a.Vector(minint(i+2, *n)-1, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1))
				e.Set(i-1, a.Get(i+1-1, i-1))
				a.Set(i+1-1, i-1, one)

				//              Compute W(i+1:n,i)
				goblas.Dsymv(Lower, toPtr((*n)-i), &one, a.Off(i+1-1, i+1-1), lda, a.Vector(i+1-1, i-1), toPtr(1), &zero, w.Vector(i+1-1, i-1), toPtr(1))
				goblas.Dgemv(Trans, toPtr((*n)-i), toPtr(i-1), &one, w.Off(i+1-1, 0), ldw, a.Vector(i+1-1, i-1), toPtr(1), &zero, w.Vector(0, i-1), toPtr(1))
				goblas.Dgemv(NoTrans, toPtr((*n)-i), toPtr(i-1), toPtrf64(-one), a.Off(i+1-1, 0), lda, w.Vector(0, i-1), toPtr(1), &one, w.Vector(i+1-1, i-1), toPtr(1))
				goblas.Dgemv(Trans, toPtr((*n)-i), toPtr(i-1), &one, a.Off(i+1-1, 0), lda, a.Vector(i+1-1, i-1), toPtr(1), &zero, w.Vector(0, i-1), toPtr(1))
				goblas.Dgemv(NoTrans, toPtr((*n)-i), toPtr(i-1), toPtrf64(-one), w.Off(i+1-1, 0), ldw, w.Vector(0, i-1), toPtr(1), &one, w.Vector(i+1-1, i-1), toPtr(1))
				goblas.Dscal(toPtr((*n)-i), tau.GetPtr(i-1), w.Vector(i+1-1, i-1), toPtr(1))
				alpha = -half * tau.Get(i-1) * goblas.Ddot(toPtr((*n)-i), w.Vector(i+1-1, i-1), toPtr(1), a.Vector(i+1-1, i-1), toPtr(1))
				goblas.Daxpy(toPtr((*n)-i), &alpha, a.Vector(i+1-1, i-1), toPtr(1), w.Vector(i+1-1, i-1), toPtr(1))
			}

		}
	}
}
