package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zget54 checks a generalized decomposition of the form
//
//          A = U*S*V'  and B = U*T* V'
//
// where ' means conjugate transpose and U and V are unitary.
//
// Specifically,
//
//   RESULT = ||( A - U*S*V', B - U*T*V' )|| / (||( A, B )||*n*ulp )
func zget54(n int, a, b, s, t, u, v *mat.CMatrix, work *mat.CVector) (result float64) {
	var cone, czero complex128
	var abnorm, one, ulp, unfl, wnorm, zero float64
	var err error

	dum := vf(1)

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	result = zero
	if n <= 0 {
		return
	}

	//     Constants
	unfl = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)

	//     compute the norm of (A,B)
	golapack.Zlacpy(Full, n, n, a, work.CMatrix(n, opts))
	golapack.Zlacpy(Full, n, n, b, work.CMatrixOff(n*n, n, opts))
	abnorm = math.Max(golapack.Zlange('1', n, 2*n, work.CMatrix(n, opts), dum), unfl)

	//     Compute W1 = A - U*S*V', and put in the array WORK(1:N*N)
	golapack.Zlacpy(Full, n, n, a, work.CMatrix(n, opts))
	if err = goblas.Zgemm(NoTrans, NoTrans, n, n, n, cone, u, s, czero, work.CMatrixOff(n*n, n, opts)); err != nil {
		panic(err)
	}

	if err = goblas.Zgemm(NoTrans, ConjTrans, n, n, n, -cone, work.CMatrixOff(n*n, n, opts), v, cone, work.CMatrix(n, opts)); err != nil {
		panic(err)
	}

	//     Compute W2 = B - U*T*V', and put in the workarray W(N*N+1:2*N*N)
	golapack.Zlacpy(Full, n, n, b, work.CMatrixOff(n*n, n, opts))
	if err = goblas.Zgemm(NoTrans, NoTrans, n, n, n, cone, u, t, czero, work.CMatrixOff(2*n*n, n, opts)); err != nil {
		panic(err)
	}

	if err = goblas.Zgemm(NoTrans, ConjTrans, n, n, n, -cone, work.CMatrixOff(2*n*n, n, opts), v, cone, work.CMatrixOff(n*n, n, opts)); err != nil {
		panic(err)
	}

	//     Compute norm(W)/ ( ulp*norm((A,B)) )
	wnorm = golapack.Zlange('1', n, 2*n, work.CMatrix(n, opts), dum)
	//
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
