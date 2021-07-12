package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zget54 checks a generalized decomposition of the form
//
//          A = U*S*V'  and B = U*T* V'
//
// where ' means conjugate transpose and U and V are unitary.
//
// Specifically,
//
//   RESULT = ||( A - U*S*V', B - U*T*V' )|| / (||( A, B )||*n*ulp )
func Zget54(n *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, s *mat.CMatrix, lds *int, t *mat.CMatrix, ldt *int, u *mat.CMatrix, ldu *int, v *mat.CMatrix, ldv *int, work *mat.CVector, result *float64) {
	var cone, czero complex128
	var abnorm, one, ulp, unfl, wnorm, zero float64
	var err error
	_ = err

	dum := vf(1)

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	(*result) = zero
	if (*n) <= 0 {
		return
	}

	//     Constants
	unfl = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)

	//     compute the norm of (A,B)
	golapack.Zlacpy('F', n, n, a, lda, work.CMatrix(*n, opts), n)
	golapack.Zlacpy('F', n, n, b, ldb, work.CMatrixOff((*n)*(*n), *n, opts), n)
	abnorm = math.Max(golapack.Zlange('1', n, toPtr(2*(*n)), work.CMatrix(*n, opts), n, dum), unfl)

	//     Compute W1 = A - U*S*V', and put in the array WORK(1:N*N)
	golapack.Zlacpy(' ', n, n, a, lda, work.CMatrix(*n, opts), n)
	err = goblas.Zgemm(NoTrans, NoTrans, *n, *n, *n, cone, u, s, czero, work.CMatrixOff((*n)*(*n), *n, opts))

	err = goblas.Zgemm(NoTrans, ConjTrans, *n, *n, *n, -cone, work.CMatrixOff((*n)*(*n), *n, opts), v, cone, work.CMatrix(*n, opts))

	//     Compute W2 = B - U*T*V', and put in the workarray W(N*N+1:2*N*N)
	golapack.Zlacpy(' ', n, n, b, ldb, work.CMatrixOff((*n)*(*n), *n, opts), n)
	err = goblas.Zgemm(NoTrans, NoTrans, *n, *n, *n, cone, u, t, czero, work.CMatrixOff(2*(*n)*(*n), *n, opts))

	err = goblas.Zgemm(NoTrans, ConjTrans, *n, *n, *n, -cone, work.CMatrixOff(2*(*n)*(*n), *n, opts), v, cone, work.CMatrixOff((*n)*(*n), *n, opts))

	//     Compute norm(W)/ ( ulp*norm((A,B)) )
	wnorm = golapack.Zlange('1', n, toPtr(2*(*n)), work.CMatrix(*n, opts), n, dum)
	//
	if abnorm > wnorm {
		(*result) = (wnorm / abnorm) / (2 * float64(*n) * ulp)
	} else {
		if abnorm < one {
			(*result) = (math.Min(wnorm, 2*float64(*n)*abnorm) / abnorm) / (2 * float64(*n) * ulp)
		} else {
			(*result) = math.Min(wnorm/abnorm, float64(2*(*n))) / (2 * float64(*n) * ulp)
		}
	}
}
