package lin

import (
	"golinalg/golapack/gltest/matgen"
	"golinalg/mat"
	"math"
)

// Zlatsp generates a special test matrix for the complex symmetric
// (indefinite) factorization for packed matrices.  The pivot blocks of
// the generated matrix will be in the following order:
//    2x2 pivot block, non diagonalizable
//    1x1 pivot block
//    2x2 pivot block, diagonalizable
//    (cycle repeats)
// A row interchange is required for each non-diagonalizable 2x2 block.
func Zlatsp(uplo byte, n *int, x *mat.CVector, iseed *[]int) {
	var a, b, c, eye, r complex128
	var alpha, alpha3, beta float64
	var j, jj, n5 int

	eye = (0.0 + 1.0*1i)

	//     Initialize constants
	alpha = (1. + math.Sqrt(17)) / 8.
	beta = alpha - 1./1000.
	alpha3 = alpha * alpha * alpha

	//     Fill the matrix with zeros.
	for j = 1; j <= (*n)*((*n)+1)/2; j++ {
		x.Set(j-1, 0.0)
	}

	//     UPLO = 'U':  Upper triangular storage
	if uplo == 'U' {
		n5 = (*n) / 5
		n5 = (*n) - 5*n5 + 1

		jj = (*n) * ((*n) + 1) / 2
		for j = (*n); j >= n5; j -= 5 {
			a = complex(alpha3, 0) * matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed)
			b = matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed) / complex(alpha, 0)
			c = a - 2.*b*eye
			r = c / complex(beta, 0)
			x.Set(jj-1, a)
			x.Set(jj-2-1, b)
			jj = jj - j
			x.Set(jj-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			x.Set(jj-1-1, r)
			jj = jj - (j - 1)
			x.Set(jj-1, c)
			jj = jj - (j - 2)
			x.Set(jj-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			jj = jj - (j - 3)
			x.Set(jj-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			if x.GetMag(jj+(j-3)-1) > x.GetMag(jj-1) {
				x.Set(jj+(j-4)-1, 2.0*x.Get(jj+(j-3)-1))
			} else {
				x.Set(jj+(j-4)-1, 2.0*x.Get(jj-1))
			}
			jj = jj - (j - 4)
		}

		//        Clean-up for N not a multiple of 5.
		j = n5 - 1
		if j > 2 {
			a = complex(alpha3, 0) * matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed)
			b = matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed) / complex(alpha, 0)
			c = a - 2.*b*eye
			r = c / complex(beta, 0)
			x.Set(jj-1, a)
			x.Set(jj-2-1, b)
			jj = jj - j
			x.Set(jj-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			x.Set(jj-1-1, r)
			jj = jj - (j - 1)
			x.Set(jj-1, c)
			jj = jj - (j - 2)
			j = j - 3
		}
		if j > 1 {
			x.Set(jj-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			x.Set(jj-j-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			if x.GetMag(jj-1) > x.GetMag(jj-j-1) {
				x.Set(jj-1-1, 2.0*x.Get(jj-1))
			} else {
				x.Set(jj-1-1, 2.0*x.Get(jj-j-1))
			}
			jj = jj - j - (j - 1)
			j = j - 2
		} else if j == 1 {
			x.Set(jj-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			j = j - 1
		}

		//     UPLO = 'L':  Lower triangular storage
	} else {
		n5 = (*n) / 5
		n5 = n5 * 5

		jj = 1
		for j = 1; j <= n5; j += 5 {
			a = complex(alpha3, 0) * matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed)
			b = matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed) / complex(alpha, 0)
			c = a - 2.*b*eye
			r = c / complex(beta, 0)
			x.Set(jj-1, a)
			x.Set(jj+2-1, b)
			jj = jj + ((*n) - j + 1)
			x.Set(jj-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			x.Set(jj+1-1, r)
			jj = jj + ((*n) - j)
			x.Set(jj-1, c)
			jj = jj + ((*n) - j - 1)
			x.Set(jj-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			jj = jj + ((*n) - j - 2)
			x.Set(jj-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			if x.GetMag(jj-((*n)-j-2)-1) > x.GetMag(jj-1) {
				x.Set(jj-((*n)-j-2)+1-1, 2.0*x.Get(jj-((*n)-j-2)-1))
			} else {
				x.Set(jj-((*n)-j-2)+1-1, 2.0*x.Get(jj-1))
			}
			jj = jj + ((*n) - j - 3)
		}

		//        Clean-up for N not a multiple of 5.
		j = n5 + 1
		if j < (*n)-1 {
			a = complex(alpha3, 0) * matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed)
			b = matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed) / complex(alpha, 0)
			c = a - 2.*b*eye
			r = c / complex(beta, 0)
			x.Set(jj-1, a)
			x.Set(jj+2-1, b)
			jj = jj + ((*n) - j + 1)
			x.Set(jj-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			x.Set(jj+1-1, r)
			jj = jj + ((*n) - j)
			x.Set(jj-1, c)
			jj = jj + ((*n) - j - 1)
			j = j + 3
		}
		if j < (*n) {
			x.Set(jj-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			x.Set(jj+((*n)-j+1)-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			if x.GetMag(jj-1) > x.GetMag(jj+((*n)-j+1)-1) {
				x.Set(jj+1-1, 2.0*x.Get(jj-1))
			} else {
				x.Set(jj+1-1, 2.0*x.Get(jj+((*n)-j+1)-1))
			}
			jj = jj + ((*n) - j + 1) + ((*n) - j)
			j = j + 2
		} else if j == (*n) {
			x.Set(jj-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			jj = jj + ((*n) - j + 1)
			j = j + 1
		}
	}
}
