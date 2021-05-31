package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgebd2 reduces a complex general m by n matrix A to upper or lower
// real bidiagonal form B by a unitary transformation: Q**H * A * P = B.
//
// If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.
func Zgebd2(m, n *int, a *mat.CMatrix, lda *int, d, e *mat.Vector, tauq, taup, work *mat.CVector, info *int) {
	var alpha, one, zero complex128
	var i int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Test the input parameters
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *m) {
		(*info) = -4
	}
	if (*info) < 0 {
		gltest.Xerbla([]byte("ZGEBD2"), -(*info))
		return
	}

	if (*m) >= (*n) {
		//        Reduce to upper bidiagonal form
		for i = 1; i <= (*n); i++ {
			//           Generate elementary reflector H(i) to annihilate A(i+1:m,i)
			alpha = a.Get(i-1, i-1)
			Zlarfg(toPtr((*m)-i+1), &alpha, a.CVector(minint(i+1, *m)-1, i-1), func() *int { y := 1; return &y }(), tauq.GetPtr(i-1))
			d.Set(i-1, real(alpha))
			a.Set(i-1, i-1, one)

			//           Apply H(i)**H to A(i:m,i+1:n) from the left
			if i < (*n) {
				Zlarf('L', toPtr((*m)-i+1), toPtr((*n)-i), a.CVector(i-1, i-1), func() *int { y := 1; return &y }(), toPtrc128(tauq.GetConj(i-1)), a.Off(i-1, i+1-1), lda, work)
			}
			a.SetRe(i-1, i-1, d.Get(i-1))

			if i < (*n) {
				//              Generate elementary reflector G(i) to annihilate
				//              A(i,i+2:n)
				Zlacgv(toPtr((*n)-i), a.CVector(i-1, i+1-1), lda)
				alpha = a.Get(i-1, i+1-1)
				Zlarfg(toPtr((*n)-i), &alpha, a.CVector(i-1, minint(i+2, *n)-1), lda, taup.GetPtr(i-1))
				e.Set(i-1, real(alpha))
				a.Set(i-1, i+1-1, one)

				//              Apply G(i) to A(i+1:m,i+1:n) from the right
				Zlarf('R', toPtr((*m)-i), toPtr((*n)-i), a.CVector(i-1, i+1-1), lda, taup.GetPtr(i-1), a.Off(i+1-1, i+1-1), lda, work)
				Zlacgv(toPtr((*n)-i), a.CVector(i-1, i+1-1), lda)
				a.SetRe(i-1, i+1-1, e.Get(i-1))
			} else {
				taup.Set(i-1, zero)
			}
		}
	} else {
		//        Reduce to lower bidiagonal form
		for i = 1; i <= (*m); i++ {
			//           Generate elementary reflector G(i) to annihilate A(i,i+1:n)
			Zlacgv(toPtr((*n)-i+1), a.CVector(i-1, i-1), lda)
			alpha = a.Get(i-1, i-1)
			Zlarfg(toPtr((*n)-i+1), &alpha, a.CVector(i-1, minint(i+1, *n)-1), lda, taup.GetPtr(i-1))
			d.Set(i-1, real(alpha))
			a.Set(i-1, i-1, one)

			//           Apply G(i) to A(i+1:m,i:n) from the right
			if i < (*m) {
				Zlarf('R', toPtr((*m)-i), toPtr((*n)-i+1), a.CVector(i-1, i-1), lda, taup.GetPtr(i-1), a.Off(i+1-1, i-1), lda, work)
			}
			Zlacgv(toPtr((*n)-i+1), a.CVector(i-1, i-1), lda)
			a.SetRe(i-1, i-1, d.Get(i-1))

			if i < (*m) {
				//              Generate elementary reflector H(i) to annihilate
				//              A(i+2:m,i)
				alpha = a.Get(i+1-1, i-1)
				Zlarfg(toPtr((*m)-i), &alpha, a.CVector(minint(i+2, *m)-1, i-1), func() *int { y := 1; return &y }(), tauq.GetPtr(i-1))
				e.Set(i-1, real(alpha))
				a.Set(i+1-1, i-1, one)

				//              Apply H(i)**H to A(i+1:m,i+1:n) from the left
				Zlarf('L', toPtr((*m)-i), toPtr((*n)-i), a.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), toPtrc128(tauq.GetConj(i-1)), a.Off(i+1-1, i+1-1), lda, work)
				a.SetRe(i+1-1, i-1, e.Get(i-1))
			} else {
				tauq.Set(i-1, zero)
			}
		}
	}
}
