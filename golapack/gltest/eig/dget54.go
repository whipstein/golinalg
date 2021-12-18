package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dget54 checks a generalized decomposition of the form
//
//          A = U*S*V'  and B = U*T* V'
//
// where ' means transpose and U and V are orthogonal.
//
// Specifically,
//
//  RESULT = ||( A - U*S*V', B - U*T*V' )|| / (||( A, B )||*n*ulp )
func Dget54(n int, a, b, s, t, u, v *mat.Matrix, work *mat.Vector) (result float64) {
	var abnorm, one, ulp, unfl, wnorm, zero float64
	var err error

	dum := vf(1)

	zero = 0.0
	one = 1.0

	if n <= 0 {
		return
	}

	//     Constants
	unfl = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)

	//     compute the norm of (A,B)
	golapack.Dlacpy(Full, n, n, a, work.Matrix(n, opts))
	golapack.Dlacpy(Full, n, n, b, work.Off(n*n).Matrix(n, opts))
	abnorm = math.Max(golapack.Dlange('1', n, 2*n, work.Matrix(n, opts), dum), unfl)

	//     Compute W1 = A - U*S*V', and put in the array WORK(1:N*N)
	golapack.Dlacpy(Full, n, n, a, work.Matrix(n, opts))
	if err = work.Off(n*n).Matrix(n, opts).Gemm(NoTrans, NoTrans, n, n, n, one, u, s, zero); err != nil {
		panic(err)
	}

	if err = work.Matrix(n, opts).Gemm(NoTrans, ConjTrans, n, n, n, -one, work.Off(n*n).Matrix(n, opts), v, one); err != nil {
		panic(err)
	}

	//     Compute W2 = B - U*T*V', and put in the workarray W(N*N+1:2*N*N)
	golapack.Dlacpy(Full, n, n, b, work.Off(n*n).Matrix(n, opts))
	if err = work.Off(2*n*n).Matrix(n, opts).Gemm(NoTrans, NoTrans, n, n, n, one, u, t, zero); err != nil {
		panic(err)
	}

	if err = work.Off(n*n).Matrix(n, opts).Gemm(NoTrans, ConjTrans, n, n, n, -one, work.Off(2*n*n).Matrix(n, opts), v, one); err != nil {
		panic(err)
	}

	//     Compute norm(W)/ ( ulp*norm((A,B)) )
	wnorm = golapack.Dlange('1', n, 2*n, work.Matrix(n, opts), dum)

	if abnorm > wnorm {
		result = (wnorm / abnorm) / (2 * float64(n) * ulp)
	} else {
		if abnorm < one {
			result = (math.Min(wnorm, 2*float64(n)*abnorm) / abnorm) / (2 * float64(n) * ulp)
		} else {
			result = math.Min(wnorm/abnorm, float64(2*n)) / (2 * float64(n) * ulp)
		}
	}

	return
}
