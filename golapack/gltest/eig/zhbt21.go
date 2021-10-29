package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zhbt21 generally checks a decomposition of the form
//
//         A = U S U**H
//
// where **H means conjugate transpose, A is hermitian banded, U is
// unitary, and S is diagonal (if KS=0) or symmetric
// tridiagonal (if KS=1).
//
// Specifically:
//
//         RESULT(1) = | A - U S U**H | / ( |A| n ulp ) and
//         RESULT(2) = | I - U U**H | / ( n ulp )
func zhbt21(uplo mat.MatUplo, n, ka, ks int, a *mat.CMatrix, d, e *mat.Vector, u *mat.CMatrix, work *mat.CVector, rwork, result *mat.Vector) {
	var lower bool
	var cuplo mat.MatUplo
	var cone, czero complex128
	var anorm, one, ulp, unfl, wnorm, zero float64
	var ika, j, jc, jr int
	var err error

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	zero = 0.0
	one = 1.0

	//     Constants
	result.Set(0, zero)
	result.Set(1, zero)
	if n <= 0 {
		return
	}

	ika = max(0, min(n-1, ka))

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
	anorm = math.Max(golapack.Zlanhb('1', cuplo, n, ika, a, rwork), unfl)

	//     Compute error matrix:    Error = A - U S U**H
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
				work.SetRe(j-1, zero)
			}
		} else {
			for jr = ika + 2; jr <= jc; jr++ {
				j = j + 1
				work.SetRe(j-1, zero)
			}
			for jr = min(ika, jc-1); jr >= 0; jr -= 1 {
				j = j + 1
				work.Set(j-1, a.Get(ika+1-jr-1, jc-1))
			}
		}
	}

	for j = 1; j <= n; j++ {
		if err = goblas.Zhpr(cuplo, n, -d.Get(j-1), u.CVector(0, j-1, 1), work); err != nil {
			panic(err)
		}
	}

	if n > 1 && ks == 1 {
		for j = 1; j <= n-1; j++ {
			if err = goblas.Zhpr2(cuplo, n, -e.GetCmplx(j-1), u.CVector(0, j-1, 1), u.CVector(0, j, 1), work); err != nil {
				panic(err)
			}
		}
	}
	wnorm = golapack.Zlanhp('1', cuplo, n, work, rwork)

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
	//     Compute  U U**H - I
	if err = goblas.Zgemm(NoTrans, ConjTrans, n, n, n, cone, u, u, czero, work.CMatrix(n, opts)); err != nil {
		panic(err)
	}

	for j = 1; j <= n; j++ {
		work.Set((n+1)*(j-1), work.Get((n+1)*(j-1))-cone)
	}

	result.Set(1, math.Min(golapack.Zlange('1', n, n, work.CMatrix(n, opts), rwork), float64(n))/(float64(n)*ulp))
}
