package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zgehd2 reduces a complex general matrix A to upper Hessenberg form H
// by a unitary similarity transformation:  Q**H * A * Q = H .
func Zgehd2(n, ilo, ihi *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector, info *int) {
	var alpha, one complex128
	var i int

	one = (1.0 + 0.0*1i)

	//     Test the input parameters
	(*info) = 0
	if (*n) < 0 {
		(*info) = -1
	} else if (*ilo) < 1 || (*ilo) > maxint(1, *n) {
		(*info) = -2
	} else if (*ihi) < minint(*ilo, *n) || (*ihi) > (*n) {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGEHD2"), -(*info))
		return
	}

	for i = (*ilo); i <= (*ihi)-1; i++ {
		//        Compute elementary reflector H(i) to annihilate A(i+2:ihi,i)
		alpha = a.Get(i+1-1, i-1)
		Zlarfg(toPtr((*ihi)-i), &alpha, a.CVector(minint(i+2, *n)-1, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1))
		a.Set(i+1-1, i-1, one)

		//        Apply H(i) to A(1:ihi,i+1:ihi) from the right
		Zlarf('R', ihi, toPtr((*ihi)-i), a.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1), a.Off(0, i+1-1), lda, work)

		//        Apply H(i)**H to A(i+1:ihi,i+1:n) from the left
		Zlarf('L', toPtr((*ihi)-i), toPtr((*n)-i), a.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), toPtrc128(tau.GetConj(i-1)), a.Off(i+1-1, i+1-1), lda, work)

		a.Set(i+1-1, i-1, alpha)
	}
}