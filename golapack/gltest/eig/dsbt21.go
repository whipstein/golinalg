package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dsbt21 generally checks a decomposition of the form
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
func dsbt21(uplo mat.MatUplo, n, ka, ks int, a *mat.Matrix, d, e *mat.Vector, u *mat.Matrix, work, result *mat.Vector) {
	var lower bool
	var cuplo mat.MatUplo
	var anorm, one, ulp, unfl, wnorm, zero float64
	var ika, j, jc, jr, lw int
	var err error

	zero = 0.0
	one = 1.0

	//     Constants
	result.Set(0, zero)
	result.Set(1, zero)
	if n <= 0 {
		return
	}

	ika = max(0, min(n-1, ka))
	lw = (n * (n + 1)) / 2

	if uplo == Upper {
		lower = false
		cuplo = Upper
	} else {
		lower = true
		cuplo = Lower
	}

	unfl = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)

	//     Some Error Checks
	//
	//     Do Test 1
	//
	//     Norm of A:
	anorm = math.Max(golapack.Dlansb('1', cuplo, n, ika, a, work), unfl)

	//     Compute error matrix:    Error = A - U S U**T
	//
	//     Copy A from SB to SP storage format.
	j = 0
	for jc = 1; jc <= n; jc++ {
		if lower {
			for jr = 1; jr <= min(ika+1, n+1-jc); jr++ {
				j = j + 1
				work.Set(j-1, a.Get(jr-1, jc-1))
			}
			for jr = ika + 2; jr <= n+1-jc; jr++ {
				j = j + 1
				work.Set(j-1, zero)
			}
		} else {
			for jr = ika + 2; jr <= jc; jr++ {
				j = j + 1
				work.Set(j-1, zero)
			}
			for jr = min(ika, jc-1); jr >= 0; jr-- {
				j = j + 1
				work.Set(j-1, a.Get(ika+1-jr-1, jc-1))
			}
		}
	}

	for j = 1; j <= n; j++ {
		err = work.Spr(cuplo, n, -d.Get(j-1), u.Off(0, j-1).Vector(), 1)
	}

	if n > 1 && ks == 1 {
		for j = 1; j <= n-1; j++ {
			err = work.Spr2(cuplo, n, -e.Get(j-1), u.Off(0, j-1).Vector(), 1, u.Off(0, j).Vector(), 1)
		}
	}
	wnorm = golapack.Dlansp('1', cuplo, n, work, work.Off(lw))

	if anorm > wnorm {
		result.Set(0, (wnorm/anorm)/(float64(n)*ulp))
	} else {
		if anorm < one {
			result.Set(0, (math.Min(wnorm, float64(n)*anorm)/anorm)/(float64(n)*ulp))
		} else {
			result.Set(0, math.Min(wnorm/anorm, float64(n))/(float64(n)*ulp))
		}
	}

	//     Do Test 2
	//
	//     Compute  U U**T - I
	if err = work.Matrix(n, opts).Gemm(NoTrans, ConjTrans, n, n, n, one, u, u, zero); err != nil {
		panic(err)
	}

	for j = 1; j <= n; j++ {
		work.Set((n+1)*(j-1), work.Get((n+1)*(j-1))-one)
	}

	result.Set(1, math.Min(golapack.Dlange('1', n, n, work.Matrix(n, opts), work.Off(pow(n, 2))), float64(n))/(float64(n)*ulp))
}
