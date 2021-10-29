package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dstt21 checks a decomposition of the form
//
//    A = U S U'
//
// where ' means transpose, A is symmetric tridiagonal, U is orthogonal,
// and S is diagonal (if KBAND=0) or symmetric tridiagonal (if KBAND=1).
// Two tests are performed:
//
//    RESULT(1) = | A - U S U' | / ( |A| n ulp )
//
//    RESULT(2) = | I - UU' | / ( n ulp )
func dstt21(n, kband int, ad, ae, sd, se *mat.Vector, u *mat.Matrix, work, result *mat.Vector) {
	var anorm, one, temp1, temp2, ulp, unfl, wnorm, zero float64
	var j int
	var err error

	zero = 0.0
	one = 1.0

	//     1)      Constants
	result.Set(0, zero)
	result.Set(1, zero)
	if n <= 0 {
		return
	}

	unfl = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Precision)

	//     Do Test 1
	//
	//     Copy A & Compute its 1-Norm:
	golapack.Dlaset(Full, n, n, zero, zero, work.Matrix(n, opts))

	anorm = zero
	temp1 = zero

	for j = 1; j <= n-1; j++ {
		work.Set((n+1)*(j-1), ad.Get(j-1))
		work.Set((n+1)*(j-1)+2-1, ae.Get(j-1))
		temp2 = math.Abs(ae.Get(j - 1))
		anorm = math.Max(anorm, math.Abs(ad.Get(j-1))+temp1+temp2)
		temp1 = temp2
	}

	work.Set(pow(n, 2)-1, ad.Get(n-1))
	anorm = math.Max(anorm, math.Max(math.Abs(ad.Get(n-1))+temp1, unfl))

	//     Norm of A - USU'
	for j = 1; j <= n; j++ {
		if err = goblas.Dsyr(Lower, n, -sd.Get(j-1), u.Vector(0, j-1, 1), work.Matrix(n, opts)); err != nil {
			panic(err)
		}
	}

	if n > 1 && kband == 1 {
		for j = 1; j <= n-1; j++ {
			if err = goblas.Dsyr2(Lower, n, -se.Get(j-1), u.Vector(0, j-1, 1), u.Vector(0, j, 1), work.Matrix(n, opts)); err != nil {
				panic(err)
			}
		}
	}

	wnorm = golapack.Dlansy('1', Lower, n, work.Matrix(n, opts), work.Off(pow(n, 2)))

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
	//     Compute  UU' - I
	if err = goblas.Dgemm(NoTrans, ConjTrans, n, n, n, one, u, u, zero, work.Matrix(n, opts)); err != nil {
		panic(err)
	}

	for j = 1; j <= n; j++ {
		work.Set((n+1)*(j-1), work.Get((n+1)*(j-1))-one)
	}

	result.Set(1, math.Min(float64(n), golapack.Dlange('1', n, n, work.Matrix(n, opts), work.Off(pow(n, 2))))/(float64(n)*ulp))
}
