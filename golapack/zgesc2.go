package golapack

import (
	"golinalg/goblas"
	"golinalg/mat"
)

// Zgesc2 solves a system of linear equations
//
//           A * X = scale* RHS
//
// with a general N-by-N matrix A using the LU factorization with
// complete pivoting computed by ZGETC2.
func Zgesc2(n *int, a *mat.CMatrix, lda *int, rhs *mat.CVector, ipiv, jpiv *[]int, scale *float64) {
	var temp complex128
	var bignum, eps, one, smlnum, two, zero float64
	var i, j int

	zero = 0.0
	one = 1.0
	two = 2.0

	//     Set constant to control overflow
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum) / eps
	bignum = one / smlnum
	Dlabad(&smlnum, &bignum)

	//     Apply permutations IPIV to RHS
	Zlaswp(func() *int { y := 1; return &y }(), rhs.CMatrix(*lda, opts), lda, func() *int { y := 1; return &y }(), toPtr((*n)-1), ipiv, func() *int { y := 1; return &y }())

	//     Solve for L part
	for i = 1; i <= (*n)-1; i++ {
		for j = i + 1; j <= (*n); j++ {
			rhs.Set(j-1, rhs.Get(j-1)-a.Get(j-1, i-1)*rhs.Get(i-1))
		}
	}

	//     Solve for U part
	(*scale) = one

	//     Check for scaling
	i = goblas.Izamax(n, rhs, func() *int { y := 1; return &y }())
	if two*smlnum*rhs.GetMag(i-1) > a.GetMag((*n)-1, (*n)-1) {
		temp = complex(one/two, zero) / complex(rhs.GetMag(i-1), 0)
		goblas.Zscal(n, &temp, rhs.Off(0), func() *int { y := 1; return &y }())
		(*scale) = (*scale) * real(temp)
	}
	for i = (*n); i >= 1; i -= 1 {
		temp = complex(one, zero) / a.Get(i-1, i-1)
		rhs.Set(i-1, rhs.Get(i-1)*temp)
		for j = i + 1; j <= (*n); j++ {
			rhs.Set(i-1, rhs.Get(i-1)-rhs.Get(j-1)*(a.Get(i-1, j-1)*temp))
		}
	}

	//     Apply permutations JPIV to the solution (RHS)
	Zlaswp(func() *int { y := 1; return &y }(), rhs.CMatrix(*lda, opts), lda, func() *int { y := 1; return &y }(), toPtr((*n)-1), jpiv, toPtr(-1))
}
