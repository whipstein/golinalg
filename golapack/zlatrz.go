package golapack

import (
	"golinalg/mat"
	"math/cmplx"
)

// Zlatrz factors the M-by-(M+L) complex upper trapezoidal matrix
// [ A1 A2 ] = [ A(1:M,1:M) A(1:M,N-L+1:N) ] as ( R  0 ) * Z by means
// of unitary transformations, where  Z is an (M+L)-by-(M+L) unitary
// matrix and, R and A1 are M-by-M upper triangular matrices.
func Zlatrz(m, n, l *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector) {
	var alpha, zero complex128
	var i int

	zero = (0.0 + 0.0*1i)

	//     Quick return if possible
	if (*m) == 0 {
		return
	} else if (*m) == (*n) {
		for i = 1; i <= (*n); i++ {
			tau.Set(i-1, zero)
		}
		return
	}

	for i = (*m); i >= 1; i-- {
		//        Generate elementary reflector H(i) to annihilate
		//        [ A(i,i) A(i,n-l+1:n) ]
		Zlacgv(l, a.CVector(i-1, (*n)-(*l)+1-1), lda)
		alpha = a.GetConj(i-1, i-1)
		Zlarfg(toPtr((*l)+1), &alpha, a.CVector(i-1, (*n)-(*l)+1-1), lda, tau.GetPtr(i-1))
		tau.Set(i-1, tau.GetConj(i-1))

		//        Apply H(i) to A(1:i-1,i:n) from the right
		Zlarz('R', toPtr(i-1), toPtr((*n)-i+1), l, a.CVector(i-1, (*n)-(*l)+1-1), lda, toPtrc128(tau.GetConj(i-1)), a.Off(0, i-1), lda, work)
		a.Set(i-1, i-1, cmplx.Conj(alpha))

	}
}
