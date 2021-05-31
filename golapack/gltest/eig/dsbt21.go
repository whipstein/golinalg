package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dsbt21 generally checks a decomposition of the form
//
//         A = U S U**T
//
// where **T means transpose, A is symmetric banded, U is
// orthogonal, and S is diagonal (if KS=0) or symmetric
// tridiagonal (if KS=1).
//
// Specifically:
//
//         RESULT(1) = | A - U S U**T | / ( |A| n ulp ) and
//         RESULT(2) = | I - U U**T | / ( n ulp )
func Dsbt21(uplo byte, n, ka, ks *int, a *mat.Matrix, lda *int, d, e *mat.Vector, u *mat.Matrix, ldu *int, work, result *mat.Vector) {
	var lower bool
	var cuplo byte
	var anorm, one, ulp, unfl, wnorm, zero float64
	var ika, j, jc, jr, lw int

	zero = 0.0
	one = 1.0

	//     Constants
	result.Set(0, zero)
	result.Set(1, zero)
	if (*n) <= 0 {
		return
	}

	ika = maxint(0, minint((*n)-1, *ka))
	lw = ((*n) * ((*n) + 1)) / 2

	if uplo == 'U' {
		lower = false
		cuplo = 'U'
	} else {
		lower = true
		cuplo = 'L'
	}

	unfl = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)

	//     Some Error Checks
	//
	//     Do Test 1
	//
	//     Norm of A:
	anorm = maxf64(golapack.Dlansb('1', cuplo, n, &ika, a, lda, work), unfl)

	//     Compute error matrix:    Error = A - U S U**T
	//
	//     Copy A from SB to SP storage format.
	j = 0
	for jc = 1; jc <= (*n); jc++ {
		if lower {
			for jr = 1; jr <= minint(ika+1, (*n)+1-jc); jr++ {
				j = j + 1
				work.Set(j-1, a.Get(jr-1, jc-1))
			}
			for jr = ika + 2; jr <= (*n)+1-jc; jr++ {
				j = j + 1
				work.Set(j-1, zero)
			}
		} else {
			for jr = ika + 2; jr <= jc; jr++ {
				j = j + 1
				work.Set(j-1, zero)
			}
			for jr = minint(ika, jc-1); jr >= 0; jr-- {
				j = j + 1
				work.Set(j-1, a.Get(ika+1-jr-1, jc-1))
			}
		}
	}

	for j = 1; j <= (*n); j++ {
		goblas.Dspr(mat.UploByte(cuplo), n, toPtrf64(-d.Get(j-1)), u.Vector(0, j-1), func() *int { y := 1; return &y }(), work)
	}

	if (*n) > 1 && (*ks) == 1 {
		for j = 1; j <= (*n)-1; j++ {
			goblas.Dspr2(mat.UploByte(cuplo), n, toPtrf64(-e.Get(j-1)), u.Vector(0, j-1), func() *int { y := 1; return &y }(), u.Vector(0, j+1-1), func() *int { y := 1; return &y }(), work)
		}
	}
	wnorm = golapack.Dlansp('1', cuplo, n, work, work.Off(lw+1-1))

	if anorm > wnorm {
		result.Set(0, (wnorm/anorm)/(float64(*n)*ulp))
	} else {
		if anorm < one {
			result.Set(0, (minf64(wnorm, float64(*n)*anorm)/anorm)/(float64(*n)*ulp))
		} else {
			result.Set(0, minf64(wnorm/anorm, float64(*n))/(float64(*n)*ulp))
		}
	}

	//     Do Test 2
	//
	//     Compute  U U**T - I
	goblas.Dgemm(NoTrans, ConjTrans, n, n, n, &one, u, ldu, u, ldu, &zero, work.Matrix(*n, opts), n)

	for j = 1; j <= (*n); j++ {
		work.Set(((*n)+1)*(j-1)+1-1, work.Get(((*n)+1)*(j-1)+1-1)-one)
	}

	result.Set(1, minf64(golapack.Dlange('1', n, n, work.Matrix(*n, opts), n, work.Off(int(math.Pow(float64(*n), 2))+1-1)), float64(*n))/(float64(*n)*ulp))
}
