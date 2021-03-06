package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zhst01 tests the reduction of a general matrix A to upper Hessenberg
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
func zhst01(n, ilo, ihi int, a, h, q *mat.CMatrix, work *mat.CVector, lwork int, rwork, result *mat.Vector) {
	var anorm, eps, one, ovfl, smlnum, unfl, wnorm, zero float64
	var ldwork int
	var err error

	one = 1.0
	zero = 0.0

	//     Quick return if possible
	if n <= 0 {
		result.Set(0, zero)
		result.Set(1, zero)
		return
	}

	unfl = golapack.Dlamch(SafeMinimum)
	eps = golapack.Dlamch(Precision)
	ovfl = one / unfl
	unfl, ovfl = golapack.Dlabad(unfl, ovfl)
	smlnum = unfl * float64(n) / eps

	//     Test 1:  Compute norm( A - Q*H*Q' ) / ( norm(A) * N * EPS )
	//
	//     Copy A to WORK
	ldwork = max(1, n)
	golapack.Zlacpy(Full, n, n, a, work.CMatrix(ldwork, opts))

	//     Compute Q*H
	if err = work.Off(ldwork*n).CMatrix(ldwork, opts).Gemm(NoTrans, NoTrans, n, n, n, complex(one, 0), q, h, complex(zero, 0)); err != nil {
		panic(err)
	}

	//     Compute A - Q*H*Q'
	if err = work.CMatrix(ldwork, opts).Gemm(NoTrans, ConjTrans, n, n, n, complex(-one, 0), work.Off(ldwork*n).CMatrix(ldwork, opts), q, complex(one, 0)); err != nil {
		panic(err)
	}

	anorm = math.Max(golapack.Zlange('1', n, n, a, rwork), unfl)
	wnorm = golapack.Zlange('1', n, n, work.CMatrix(ldwork, opts), rwork)

	//     Note that RESULT(1) cannot overflow and is bounded by 1/(N*EPS)
	result.Set(0, math.Min(wnorm, anorm)/math.Max(smlnum, anorm*eps)/float64(n))

	//     Test 2:  Compute norm( I - Q'*Q ) / ( N * EPS )
	result.Set(1, zunt01('C', n, n, q, work, lwork, rwork))
}
