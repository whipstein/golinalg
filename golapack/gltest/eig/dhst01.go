package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dhst01 tests the reduction of a general matrix A to upper Hessenberg
// form:  A = Q*H*Q'.  Two test ratios are computed;
//
// RESULT(1) = norm( A - Q*H*Q' ) / ( norm(A) * N * EPS )
// RESULT(2) = norm( I - Q'*Q ) / ( N * EPS )
//
// The matrix Q is assumed to be given explicitly as it would be
// following DGEHRD + DORGHR.
//
// In this version, ILO and IHI are not used and are assumed to be 1 and
// N, respectively.
func Dhst01(n, ilo, ihi *int, a *mat.Matrix, lda *int, h *mat.Matrix, ldh *int, q *mat.Matrix, ldq *int, work *mat.Vector, lwork *int, result *mat.Vector) {
	var anorm, eps, one, ovfl, smlnum, unfl, wnorm, zero float64
	var ldwork int
	var err error
	_ = err

	one = 1.0
	zero = 0.0

	//     Quick return if possible
	if (*n) <= 0 {
		result.Set(0, zero)
		result.Set(1, zero)
		return
	}

	unfl = golapack.Dlamch(SafeMinimum)
	eps = golapack.Dlamch(Precision)
	ovfl = one / unfl
	golapack.Dlabad(&unfl, &ovfl)
	smlnum = unfl * float64(*n) / eps

	//     Test 1:  Compute norm( A - Q*H*Q' ) / ( norm(A) * N * EPS )
	//
	//     Copy A to WORK
	ldwork = max(1, *n)
	golapack.Dlacpy(' ', n, n, a, lda, work.Matrix(ldwork, opts), &ldwork)

	//     Compute Q*H
	err = goblas.Dgemm(NoTrans, NoTrans, *n, *n, *n, one, q, h, zero, work.MatrixOff(ldwork*(*n), ldwork, opts))

	//     Compute A - Q*H*Q'
	err = goblas.Dgemm(NoTrans, Trans, *n, *n, *n, -one, work.MatrixOff(ldwork*(*n), ldwork, opts), q, one, work.Matrix(ldwork, opts))

	anorm = math.Max(golapack.Dlange('1', n, n, a, lda, work.Off(ldwork*(*n))), unfl)
	wnorm = golapack.Dlange('1', n, n, work.Matrix(ldwork, opts), &ldwork, work.Off(ldwork*(*n)))

	//     Note that RESULT(1) cannot overflow and is bounded by 1/(N*EPS)
	result.Set(0, math.Min(wnorm, anorm)/math.Max(smlnum, anorm*eps)/float64(*n))

	//     Test 2:  Compute norm( I - Q'*Q ) / ( N * EPS )
	Dort01('C', n, n, q, ldq, work, lwork, result.GetPtr(1))
}
