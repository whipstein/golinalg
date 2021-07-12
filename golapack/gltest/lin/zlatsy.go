package lin

import (
	"math"

	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Zlatsy generates a special test matrix for the complex symmetric
// (indefinite) factorization.  The pivot blocks of the generated matrix
// will be in the following order:
//    2x2 pivot block, non diagonalizable
//    1x1 pivot block
//    2x2 pivot block, diagonalizable
//    (cycle repeats)
// A row interchange is required for each non-diagonalizable 2x2 block.
func Zlatsy(uplo byte, n *int, x *mat.CMatrix, ldx *int, iseed *[]int) {
	var a, b, c, eye, r complex128
	var alpha, alpha3, beta float64
	var i, j, n5 int

	eye = (0.0 + 1.0*1i)

	//     Initialize constants
	alpha = (1. + math.Sqrt(17)) / 8.
	beta = alpha - 1./1000.
	alpha3 = alpha * alpha * alpha

	//     UPLO = 'U':  Upper triangular storage
	if uplo == 'U' {
		//        Fill the upper triangle of the matrix with zeros.
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= j; i++ {
				x.Set(i-1, j-1, 0.0)
			}
		}
		n5 = (*n) / 5
		n5 = (*n) - 5*n5 + 1

		for i = (*n); i >= n5; i -= 5 {
			a = complex(alpha3, 0) * matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed)
			b = matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed) / complex(alpha, 0)
			c = a - 2.*b*eye
			r = c / complex(beta, 0)
			x.Set(i-1, i-1, a)
			x.Set(i-2-1, i-1, b)
			x.Set(i-2-1, i-1-1, r)
			x.Set(i-2-1, i-2-1, c)
			x.Set(i-1-1, i-1-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			x.Set(i-3-1, i-3-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			x.Set(i-4-1, i-4-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			if x.GetMag(i-3-1, i-3-1) > x.GetMag(i-4-1, i-4-1) {
				x.Set(i-4-1, i-3-1, 2.0*x.Get(i-3-1, i-3-1))
			} else {
				x.Set(i-4-1, i-3-1, 2.0*x.Get(i-4-1, i-4-1))
			}
		}

		//        Clean-up for N not a multiple of 5.
		i = n5 - 1
		if i > 2 {
			a = complex(alpha3, 0) * matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed)
			b = matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed) / complex(alpha, 0)
			c = a - 2.*b*eye
			r = c / complex(beta, 0)
			x.Set(i-1, i-1, a)
			x.Set(i-2-1, i-1, b)
			x.Set(i-2-1, i-1-1, r)
			x.Set(i-2-1, i-2-1, c)
			x.Set(i-1-1, i-1-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			i = i - 3
		}
		if i > 1 {
			x.Set(i-1, i-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			x.Set(i-1-1, i-1-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			if x.GetMag(i-1, i-1) > x.GetMag(i-1-1, i-1-1) {
				x.Set(i-1-1, i-1, 2.0*x.Get(i-1, i-1))
			} else {
				x.Set(i-1-1, i-1, 2.0*x.Get(i-1-1, i-1-1))
			}
			i = i - 2
		} else if i == 1 {
			x.Set(i-1, i-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			i = i - 1
		}

		//     UPLO = 'L':  Lower triangular storage
	} else {
		//        Fill the lower triangle of the matrix with zeros.
		for j = 1; j <= (*n); j++ {
			for i = j; i <= (*n); i++ {
				x.Set(i-1, j-1, 0.0)
			}
		}
		n5 = (*n) / 5
		n5 = n5 * 5

		for i = 1; i <= n5; i += 5 {
			a = complex(alpha3, 0) * matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed)
			b = matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed) / complex(alpha, 0)
			c = a - 2.*b*eye
			r = c / complex(beta, 0)
			x.Set(i-1, i-1, a)
			x.Set(i+2-1, i-1, b)
			x.Set(i+2-1, i, r)
			x.Set(i+2-1, i+2-1, c)
			x.Set(i, i, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			x.Set(i+3-1, i+3-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			x.Set(i+4-1, i+4-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			if x.GetMag(i+3-1, i+3-1) > x.GetMag(i+4-1, i+4-1) {
				x.Set(i+4-1, i+3-1, 2.0*x.Get(i+3-1, i+3-1))
			} else {
				x.Set(i+4-1, i+3-1, 2.0*x.Get(i+4-1, i+4-1))
			}
		}

		//        Clean-up for N not a multiple of 5.
		i = n5 + 1
		if i < (*n)-1 {
			a = complex(alpha3, 0) * matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed)
			b = matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed) / complex(alpha, 0)
			c = a - 2.*b*eye
			r = c / complex(beta, 0)
			x.Set(i-1, i-1, a)
			x.Set(i+2-1, i-1, b)
			x.Set(i+2-1, i, r)
			x.Set(i+2-1, i+2-1, c)
			x.Set(i, i, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			i = i + 3
		}
		if i < (*n) {
			x.Set(i-1, i-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			x.Set(i, i, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			if x.GetMag(i-1, i-1) > x.GetMag(i, i) {
				x.Set(i, i-1, 2.0*x.Get(i-1, i-1))
			} else {
				x.Set(i, i-1, 2.0*x.Get(i, i))
			}
			i = i + 2
		} else if i == (*n) {
			x.Set(i-1, i-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			i = i + 1
		}
	}
}
