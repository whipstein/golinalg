package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgehd2 reduces a real general matrix A to upper Hessenberg form H by
// an orthogonal similarity transformation:  Q**T * A * Q = H .
func Dgehd2(n, ilo, ihi *int, a *mat.Matrix, lda *int, tau, work *mat.Vector, info *int) {
	var aii, one float64
	var i int

	one = 1.0

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
		gltest.Xerbla([]byte("DGEHD2"), -(*info))
		return
	}

	for i = (*ilo); i <= (*ihi)-1; i++ {
		//        Compute elementary reflector H(i) to annihilate A(i+2:ihi,i)
		Dlarfg(toPtr((*ihi)-i), a.GetPtr(i+1-1, i-1), a.Vector(minint(i+2, *n)-1, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1))
		aii = a.Get(i+1-1, i-1)
		a.Set(i+1-1, i-1, one)

		//        Apply H(i) to A(1:ihi,i+1:ihi) from the right
		Dlarf('R', ihi, toPtr((*ihi)-i), a.Vector(i+1-1, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1), a.Off(0, i+1-1), lda, work)

		//        Apply H(i) to A(i+1:ihi,i+1:n) from the left
		Dlarf('L', toPtr((*ihi)-i), toPtr((*n)-i), a.Vector(i+1-1, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1), a.Off(i+1-1, i+1-1), lda, work)

		a.Set(i+1-1, i-1, aii)
	}
}
