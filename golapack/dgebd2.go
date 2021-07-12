package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgebd2 reduces a real general m by n matrix A to upper or lower
// bidiagonal form B by an orthogonal transformation: Q**T * A * P = B.
//
// If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.
func Dgebd2(m, n *int, a *mat.Matrix, lda *int, d, e, tauq, taup, work *mat.Vector, info *int) {
	var one, zero float64
	var i int

	zero = 0.0
	one = 1.0

	//     Test the input parameters
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *m) {
		(*info) = -4
	}
	if (*info) < 0 {
		gltest.Xerbla([]byte("DGEBD2"), -(*info))
		return
	}

	if (*m) >= (*n) {
		//        Reduce to upper bidiagonal form
		for i = 1; i <= (*n); i++ {
			//           Generate elementary reflector H(i) to annihilate A(i+1:m,i)
			Dlarfg(toPtr((*m)-i+1), a.GetPtr(i-1, i-1), a.Vector(min(i+1, *m)-1, i-1), func() *int { y := 1; return &y }(), tauq.GetPtr(i-1))
			d.Set(i-1, a.Get(i-1, i-1))
			a.Set(i-1, i-1, one)

			//           Apply H(i) to A(i:m,i+1:n) from the left
			if i < (*n) {
				Dlarf('L', toPtr((*m)-i+1), toPtr((*n)-i), a.Vector(i-1, i-1), func() *int { y := 1; return &y }(), tauq.GetPtr(i-1), a.Off(i-1, i), lda, work)
			}
			a.Set(i-1, i-1, d.Get(i-1))

			if i < (*n) {
				//              Generate elementary reflector G(i) to annihilate
				//              A(i,i+2:n)
				Dlarfg(toPtr((*n)-i), a.GetPtr(i-1, i), a.Vector(i-1, min(i+2, *n)-1), lda, taup.GetPtr(i-1))
				e.Set(i-1, a.Get(i-1, i))
				a.Set(i-1, i, one)

				//              Apply G(i) to A(i+1:m,i+1:n) from the right
				Dlarf('R', toPtr((*m)-i), toPtr((*n)-i), a.Vector(i-1, i), lda, taup.GetPtr(i-1), a.Off(i, i), lda, work)
				a.Set(i-1, i, e.Get(i-1))
			} else {
				taup.Set(i-1, zero)
			}
		}
	} else {
		//        Reduce to lower bidiagonal form
		for i = 1; i <= (*m); i++ {
			//           Generate elementary reflector G(i) to annihilate A(i,i+1:n)
			Dlarfg(toPtr((*n)-i+1), a.GetPtr(i-1, i-1), a.Vector(i-1, min(i+1, *n)-1), lda, taup.GetPtr(i-1))
			d.Set(i-1, a.Get(i-1, i-1))
			a.Set(i-1, i-1, one)

			//           Apply G(i) to A(i+1:m,i:n) from the right
			if i < (*m) {
				Dlarf('R', toPtr((*m)-i), toPtr((*n)-i+1), a.Vector(i-1, i-1), lda, taup.GetPtr(i-1), a.Off(i, i-1), lda, work)
			}
			a.Set(i-1, i-1, d.Get(i-1))

			if i < (*m) {
				//              Generate elementary reflector H(i) to annihilate
				//              A(i+2:m,i)
				Dlarfg(toPtr((*m)-i), a.GetPtr(i, i-1), a.Vector(min(i+2, *m)-1, i-1), func() *int { y := 1; return &y }(), tauq.GetPtr(i-1))
				e.Set(i-1, a.Get(i, i-1))
				a.Set(i, i-1, one)

				//              Apply H(i) to A(i+1:m,i+1:n) from the left
				Dlarf('L', toPtr((*m)-i), toPtr((*n)-i), a.Vector(i, i-1), func() *int { y := 1; return &y }(), tauq.GetPtr(i-1), a.Off(i, i), lda, work)
				a.Set(i, i-1, e.Get(i-1))
			} else {
				tauq.Set(i-1, zero)
			}
		}
	}
}
