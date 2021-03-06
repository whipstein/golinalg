package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dgesc2 solves a system of linear equations
//
//           A * X = scale* RHS
//
// with a general N-by-N matrix A using the LU factorization with
// complete pivoting computed by DGETC2.
func Dgesc2(n int, a *mat.Matrix, rhs *mat.Vector, ipiv, jpiv *[]int) (scale float64) {
	var bignum, eps, one, smlnum, temp, two float64
	var i, j int

	one = 1.0
	two = 2.0

	//      Set constant to control overflow
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum) / eps
	bignum = one / smlnum
	smlnum, bignum = Dlabad(smlnum, bignum)

	//     Apply permutations IPIV to RHS
	Dlaswp(1, rhs.Matrix(a.Rows, opts), 1, n-1, *ipiv, 1)

	//     Solve for L part
	for i = 1; i <= n-1; i++ {
		for j = i + 1; j <= n; j++ {
			rhs.Set(j-1, rhs.Get(j-1)-a.Get(j-1, i-1)*rhs.Get(i-1))
		}
	}

	//     Solve for U part
	scale = one

	//     Check for scaling
	i = rhs.Iamax(n, 1)
	if two*smlnum*math.Abs(rhs.Get(i-1)) > math.Abs(a.Get(n-1, n-1)) {
		temp = (one / two) / math.Abs(rhs.Get(i-1))
		rhs.Scal(n, temp, 1)
		scale = scale * temp
	}

	for i = n; i >= 1; i-- {
		temp = one / a.Get(i-1, i-1)
		rhs.Set(i-1, rhs.Get(i-1)*temp)
		for j = i + 1; j <= n; j++ {
			rhs.Set(i-1, rhs.Get(i-1)-rhs.Get(j-1)*(a.Get(i-1, j-1)*temp))
		}
	}

	//     Apply permutations JPIV to the solution (RHS)
	Dlaswp(1, rhs.Matrix(a.Rows, opts), 1, n-1, *jpiv, -1)

	return
}
