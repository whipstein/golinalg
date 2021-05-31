package eig

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zhst01 tests the reduction of a general matrix A to upper Hessenberg
// form:  A = Q*H*Q'.  Two test ratios are computed;
//
// RESULT(1) = norm( A - Q*H*Q' ) / ( norm(A) * N * EPS )
// RESULT(2) = norm( I - Q'*Q ) / ( N * EPS )
//
// The matrix Q is assumed to be given explicitly as it would be
// following ZGEHRD + ZUNGHR.
//
// In this version, ILO and IHI are not used, but they could be used
// to save some work if this is desired.
func Zhst01(n, ilo, ihi *int, a *mat.CMatrix, lda *int, h *mat.CMatrix, ldh *int, q *mat.CMatrix, ldq *int, work *mat.CVector, lwork *int, rwork, result *mat.Vector) {
	var anorm, eps, one, ovfl, smlnum, unfl, wnorm, zero float64
	var ldwork int

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
	ldwork = maxint(1, *n)
	golapack.Zlacpy(' ', n, n, a, lda, work.CMatrix(ldwork, opts), &ldwork)

	//     Compute Q*H
	goblas.Zgemm(NoTrans, NoTrans, n, n, n, toPtrc128(complex(one, 0)), q, ldq, h, ldh, toPtrc128(complex(zero, 0)), work.CMatrixOff(ldwork*(*n)+1-1, ldwork, opts), &ldwork)

	//     Compute A - Q*H*Q'
	goblas.Zgemm(NoTrans, ConjTrans, n, n, n, toPtrc128(complex(-one, 0)), work.CMatrixOff(ldwork*(*n)+1-1, ldwork, opts), &ldwork, q, ldq, toPtrc128(complex(one, 0)), work.CMatrix(ldwork, opts), &ldwork)

	anorm = maxf64(golapack.Zlange('1', n, n, a, lda, rwork), unfl)
	wnorm = golapack.Zlange('1', n, n, work.CMatrix(ldwork, opts), &ldwork, rwork)

	//     Note that RESULT(1) cannot overflow and is bounded by 1/(N*EPS)
	result.Set(0, minf64(wnorm, anorm)/maxf64(smlnum, anorm*eps)/float64(*n))

	//     Test 2:  Compute norm( I - Q'*Q ) / ( N * EPS )
	Zunt01('C', n, n, q, ldq, work, lwork, rwork, result.GetPtr(1))
}
