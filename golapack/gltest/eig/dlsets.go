package eig

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dlsets tests DGGLSE - a subroutine for solving linear equality
// constrained least square problem (LSE).
func Dlsets(m, p, n *int, a, af *mat.Matrix, lda *int, b, bf *mat.Matrix, ldb *int, c, cf, d, df, x, work *mat.Vector, lwork *int, rwork, result *mat.Vector) {
	var info int

	//     Copy the matrices A and B to the arrays AF and BF,
	//     and the vectors C and D to the arrays CF and DF,
	golapack.Dlacpy('F', m, n, a, lda, af, lda)
	golapack.Dlacpy('F', p, n, b, ldb, bf, ldb)
	goblas.Dcopy(*m, c, 1, cf, 1)
	goblas.Dcopy(*p, d, 1, df, 1)

	//     Solve LSE problem
	golapack.Dgglse(m, n, p, af, lda, bf, ldb, cf, df, x, work, lwork, &info)

	//     Test the residual for the solution of LSE
	//
	//     Compute RESULT(1) = norm( A*x - c ) / norm(A)*norm(X)*EPS
	goblas.Dcopy(*m, c, 1, cf, 1)
	goblas.Dcopy(*p, d, 1, df, 1)
	Dget02('N', m, n, func() *int { y := 1; return &y }(), a, lda, x.Matrix(*n, opts), n, cf.Matrix(*m, opts), m, rwork, result.GetPtr(0))

	//     Compute result(2) = norm( B*x - d ) / norm(B)*norm(X)*EPS
	Dget02('N', p, n, func() *int { y := 1; return &y }(), b, ldb, x.Matrix(*n, opts), n, df.Matrix(*p, opts), p, rwork, result.GetPtr(1))
}
