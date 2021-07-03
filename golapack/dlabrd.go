package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dlabrd reduces the first NB rows and columns of a real general
// m by n matrix A to upper or lower bidiagonal form by an orthogonal
// transformation Q**T * A * P, and returns the matrices X and Y which
// are needed to apply the transformation to the unreduced part of A.
//
// If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower
// bidiagonal form.
//
// This is an auxiliary routine called by DGEBRD
func Dlabrd(m, n, nb *int, a *mat.Matrix, lda *int, d, e, tauq, taup *mat.Vector, x *mat.Matrix, ldx *int, y *mat.Matrix, ldy *int) {
	var one, zero float64
	var i int
	var err error
	_ = err

	zero = 0.0
	one = 1.0

	//     Quick return if possible
	if (*m) <= 0 || (*n) <= 0 {
		return
	}

	if (*m) >= (*n) {
		//        Reduce to upper bidiagonal form
		for i = 1; i <= (*nb); i++ {
			//           Update A(i:m,i)
			err = goblas.Dgemv(NoTrans, (*m)-i+1, i-1, -one, a.Off(i-1, 0), *lda, y.Vector(i-1, 0), *ldy, one, a.Vector(i-1, i-1), 1)
			err = goblas.Dgemv(NoTrans, (*m)-i+1, i-1, -one, x.Off(i-1, 0), *ldx, a.Vector(0, i-1), 1, one, a.Vector(i-1, i-1), 1)

			//           Generate reflection Q(i) to annihilate A(i+1:m,i)
			Dlarfg(toPtr((*m)-i+1), a.GetPtr(i-1, i-1), a.Vector(minint(i+1, *m)-1, i-1), func() *int { y := 1; return &y }(), tauq.GetPtr(i-1))
			d.Set(i-1, a.Get(i-1, i-1))
			if i < (*n) {
				a.Set(i-1, i-1, one)

				//              Compute Y(i+1:n,i)
				err = goblas.Dgemv(Trans, (*m)-i+1, (*n)-i, one, a.Off(i-1, i+1-1), *lda, a.Vector(i-1, i-1), 1, zero, y.Vector(i+1-1, i-1), 1)
				err = goblas.Dgemv(Trans, (*m)-i+1, i-1, one, a.Off(i-1, 0), *lda, a.Vector(i-1, i-1), 1, zero, y.Vector(0, i-1), 1)
				err = goblas.Dgemv(NoTrans, (*n)-i, i-1, -one, y.Off(i+1-1, 0), *ldy, y.Vector(0, i-1), 1, one, y.Vector(i+1-1, i-1), 1)
				err = goblas.Dgemv(Trans, (*m)-i+1, i-1, one, x.Off(i-1, 0), *ldx, a.Vector(i-1, i-1), 1, zero, y.Vector(0, i-1), 1)
				err = goblas.Dgemv(Trans, i-1, (*n)-i, -one, a.Off(0, i+1-1), *lda, y.Vector(0, i-1), 1, one, y.Vector(i+1-1, i-1), 1)
				goblas.Dscal((*n)-i, tauq.Get(i-1), y.Vector(i+1-1, i-1), 1)

				//              Update A(i,i+1:n)
				err = goblas.Dgemv(NoTrans, (*n)-i, i, -one, y.Off(i+1-1, 0), *ldy, a.Vector(i-1, 0), *lda, one, a.Vector(i-1, i+1-1), *lda)
				err = goblas.Dgemv(Trans, i-1, (*n)-i, -one, a.Off(0, i+1-1), *lda, x.Vector(i-1, 0), *ldx, one, a.Vector(i-1, i+1-1), *lda)

				//              Generate reflection P(i) to annihilate A(i,i+2:n)
				Dlarfg(toPtr((*n)-i), a.GetPtr(i-1, i+1-1), a.Vector(i-1, minint(i+2, *n)-1), lda, taup.GetPtr(i-1))
				e.Set(i-1, a.Get(i-1, i+1-1))
				a.Set(i-1, i+1-1, one)

				//              Compute X(i+1:m,i)
				err = goblas.Dgemv(NoTrans, (*m)-i, (*n)-i, one, a.Off(i+1-1, i+1-1), *lda, a.Vector(i-1, i+1-1), *lda, zero, x.Vector(i+1-1, i-1), 1)
				err = goblas.Dgemv(Trans, (*n)-i, i, one, y.Off(i+1-1, 0), *ldy, a.Vector(i-1, i+1-1), *lda, zero, x.Vector(0, i-1), 1)
				err = goblas.Dgemv(NoTrans, (*m)-i, i, -one, a.Off(i+1-1, 0), *lda, x.Vector(0, i-1), 1, one, x.Vector(i+1-1, i-1), 1)
				err = goblas.Dgemv(NoTrans, i-1, (*n)-i, one, a.Off(0, i+1-1), *lda, a.Vector(i-1, i+1-1), *lda, zero, x.Vector(0, i-1), 1)
				err = goblas.Dgemv(NoTrans, (*m)-i, i-1, -one, x.Off(i+1-1, 0), *ldx, x.Vector(0, i-1), 1, one, x.Vector(i+1-1, i-1), 1)
				goblas.Dscal((*m)-i, taup.Get(i-1), x.Vector(i+1-1, i-1), 1)
			}
		}
	} else {
		//        Reduce to lower bidiagonal form
		for i = 1; i <= (*nb); i++ {
			//           Update A(i,i:n)
			err = goblas.Dgemv(NoTrans, (*n)-i+1, i-1, -one, y.Off(i-1, 0), *ldy, a.Vector(i-1, 0), *lda, one, a.Vector(i-1, i-1), *lda)
			err = goblas.Dgemv(Trans, i-1, (*n)-i+1, -one, a.Off(0, i-1), *lda, x.Vector(i-1, 0), *ldx, one, a.Vector(i-1, i-1), *lda)

			//           Generate reflection P(i) to annihilate A(i,i+1:n)
			Dlarfg(toPtr((*n)-i+1), a.GetPtr(i-1, i-1), a.Vector(i-1, minint(i+1, *n)-1), lda, taup.GetPtr(i-1))
			d.Set(i-1, a.Get(i-1, i-1))
			if i < (*m) {
				a.Set(i-1, i-1, one)

				//              Compute X(i+1:m,i)
				err = goblas.Dgemv(NoTrans, (*m)-i, (*n)-i+1, one, a.Off(i+1-1, i-1), *lda, a.Vector(i-1, i-1), *lda, zero, x.Vector(i+1-1, i-1), 1)
				err = goblas.Dgemv(Trans, (*n)-i+1, i-1, one, y.Off(i-1, 0), *ldy, a.Vector(i-1, i-1), *lda, zero, x.Vector(0, i-1), 1)
				err = goblas.Dgemv(NoTrans, (*m)-i, i-1, -one, a.Off(i+1-1, 0), *lda, x.Vector(0, i-1), 1, one, x.Vector(i+1-1, i-1), 1)
				err = goblas.Dgemv(NoTrans, i-1, (*n)-i+1, one, a.Off(0, i-1), *lda, a.Vector(i-1, i-1), *lda, zero, x.Vector(0, i-1), 1)
				err = goblas.Dgemv(NoTrans, (*m)-i, i-1, -one, x.Off(i+1-1, 0), *ldx, x.Vector(0, i-1), 1, one, x.Vector(i+1-1, i-1), 1)
				goblas.Dscal((*m)-i, taup.Get(i-1), x.Vector(i+1-1, i-1), 1)

				//              Update A(i+1:m,i)
				err = goblas.Dgemv(NoTrans, (*m)-i, i-1, -one, a.Off(i+1-1, 0), *lda, y.Vector(i-1, 0), *ldy, one, a.Vector(i+1-1, i-1), 1)
				err = goblas.Dgemv(NoTrans, (*m)-i, i, -one, x.Off(i+1-1, 0), *ldx, a.Vector(0, i-1), 1, one, a.Vector(i+1-1, i-1), 1)

				//              Generate reflection Q(i) to annihilate A(i+2:m,i)
				Dlarfg(toPtr((*m)-i), a.GetPtr(i+1-1, i-1), a.Vector(minint(i+2, *m)-1, i-1), func() *int { y := 1; return &y }(), tauq.GetPtr(i-1))
				e.Set(i-1, a.Get(i+1-1, i-1))
				a.Set(i+1-1, i-1, one)

				//              Compute Y(i+1:n,i)
				err = goblas.Dgemv(Trans, (*m)-i, (*n)-i, one, a.Off(i+1-1, i+1-1), *lda, a.Vector(i+1-1, i-1), 1, zero, y.Vector(i+1-1, i-1), 1)
				err = goblas.Dgemv(Trans, (*m)-i, i-1, one, a.Off(i+1-1, 0), *lda, a.Vector(i+1-1, i-1), 1, zero, y.Vector(0, i-1), 1)
				err = goblas.Dgemv(NoTrans, (*n)-i, i-1, -one, y.Off(i+1-1, 0), *ldy, y.Vector(0, i-1), 1, one, y.Vector(i+1-1, i-1), 1)
				err = goblas.Dgemv(Trans, (*m)-i, i, one, x.Off(i+1-1, 0), *ldx, a.Vector(i+1-1, i-1), 1, zero, y.Vector(0, i-1), 1)
				err = goblas.Dgemv(Trans, i, (*n)-i, -one, a.Off(0, i+1-1), *lda, y.Vector(0, i-1), 1, one, y.Vector(i+1-1, i-1), 1)
				goblas.Dscal((*n)-i, tauq.Get(i-1), y.Vector(i+1-1, i-1), 1)
			}
		}
	}
}
