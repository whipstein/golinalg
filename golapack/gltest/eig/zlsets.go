package eig

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zlsets tests ZGGLSE - a subroutine for solving linear equality
// constrained least square problem (LSE).
func Zlsets(m, p, n *int, a, af *mat.CMatrix, lda *int, b, bf *mat.CMatrix, ldb *int, c, cf, d, df, x, work *mat.CVector, lwork *int, rwork, result *mat.Vector) {
	var info int

	//     Copy the matrices A and B to the arrays AF and BF,
	//     and the vectors C and D to the arrays CF and DF,
	golapack.Zlacpy('F', m, n, a, lda, af, lda)
	golapack.Zlacpy('F', p, n, b, ldb, bf, ldb)
	goblas.Zcopy(m, c, func() *int { y := 1; return &y }(), cf, func() *int { y := 1; return &y }())
	goblas.Zcopy(p, d, func() *int { y := 1; return &y }(), df, func() *int { y := 1; return &y }())

	//     Solve LSE problem
	golapack.Zgglse(m, n, p, af, lda, bf, ldb, cf, df, x, work, lwork, &info)

	//     Test the residual for the solution of LSE
	//
	//     Compute RESULT(1) = norm( A*x - c ) / norm(A)*norm(X)*EPS
	goblas.Zcopy(m, c, func() *int { y := 1; return &y }(), cf, func() *int { y := 1; return &y }())
	goblas.Zcopy(p, d, func() *int { y := 1; return &y }(), df, func() *int { y := 1; return &y }())
	Zget02('N', m, n, func() *int { y := 1; return &y }(), a, lda, x.CMatrix(*n, opts), n, cf.CMatrix(*m, opts), m, rwork, result.GetPtr(0))

	//     Compute result(2) = norm( B*x - d ) / norm(B)*norm(X)*EPS
	Zget02('N', p, n, func() *int { y := 1; return &y }(), b, ldb, x.CMatrix(*n, opts), n, df.CMatrix(*p, opts), p, rwork, result.GetPtr(1))
}
