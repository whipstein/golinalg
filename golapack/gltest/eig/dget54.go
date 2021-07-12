package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
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
func Dget54(n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, s *mat.Matrix, lds *int, t *mat.Matrix, ldt *int, u *mat.Matrix, ldu *int, v *mat.Matrix, ldv *int, work *mat.Vector, result *float64) {
	var abnorm, one, ulp, unfl, wnorm, zero float64
	var err error
	_ = err

	dum := vf(1)

	zero = 0.0
	one = 1.0

	(*result) = zero
	if (*n) <= 0 {
		return
	}

	//     Constants
	unfl = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)

	//     compute the norm of (A,B)
	golapack.Dlacpy('F', n, n, a, lda, work.Matrix(*n, opts), n)
	golapack.Dlacpy('F', n, n, b, ldb, work.MatrixOff((*n)*(*n), *n, opts), n)
	abnorm = math.Max(golapack.Dlange('1', n, toPtr(2*(*n)), work.Matrix(*n, opts), n, dum), unfl)

	//     Compute W1 = A - U*S*V', and put in the array WORK(1:N*N)
	golapack.Dlacpy(' ', n, n, a, lda, work.Matrix(*n, opts), n)
	err = goblas.Dgemm(NoTrans, NoTrans, *n, *n, *n, one, u, s, zero, work.MatrixOff((*n)*(*n), *n, opts))

	err = goblas.Dgemm(NoTrans, ConjTrans, *n, *n, *n, -one, work.MatrixOff((*n)*(*n), *n, opts), v, one, work.Matrix(*n, opts))

	//     Compute W2 = B - U*T*V', and put in the workarray W(N*N+1:2*N*N)
	golapack.Dlacpy(' ', n, n, b, ldb, work.MatrixOff((*n)*(*n), *n, opts), n)
	err = goblas.Dgemm(NoTrans, NoTrans, *n, *n, *n, one, u, t, zero, work.MatrixOff(2*(*n)*(*n), *n, opts))

	err = goblas.Dgemm(NoTrans, ConjTrans, *n, *n, *n, -one, work.MatrixOff(2*(*n)*(*n), *n, opts), v, one, work.MatrixOff((*n)*(*n), *n, opts))

	//     Compute norm(W)/ ( ulp*norm((A,B)) )
	wnorm = golapack.Dlange('1', n, toPtr(2*(*n)), work.Matrix(*n, opts), n, dum)

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
