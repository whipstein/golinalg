package eig

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dlsets tests DGGLSE - a subroutine for solving linear equality
// constrained least square problem (LSE).
func dlsets(m, p, n int, a, af, b, bf *mat.Matrix, c, cf, d, df, x, work *mat.Vector, lwork int, rwork, result *mat.Vector) {
	var err error

	//     Copy the matrices A and B to the arrays AF and BF,
	//     and the vectors C and D to the arrays CF and DF,
	golapack.Dlacpy(Full, m, n, a, af)
	golapack.Dlacpy(Full, p, n, b, bf)
	goblas.Dcopy(m, c.Off(0, 1), cf.Off(0, 1))
	goblas.Dcopy(p, d.Off(0, 1), df.Off(0, 1))

	//     Solve LSE problem
	if _, err = golapack.Dgglse(m, n, p, af, bf, cf, df, x, work, lwork); err != nil {
		panic(err)
	}

	//     Test the residual for the solution of LSE
	//
	//     Compute RESULT(1) = norm( A*x - c ) / norm(A)*norm(X)*EPS
	goblas.Dcopy(m, c.Off(0, 1), cf.Off(0, 1))
	goblas.Dcopy(p, d.Off(0, 1), df.Off(0, 1))
	result.Set(0, dget02(NoTrans, m, n, 1, a, x.Matrix(n, opts), cf.Matrix(m, opts), rwork))

	//     Compute result(2) = norm( B*x - d ) / norm(B)*norm(X)*EPS
	result.Set(1, dget02(NoTrans, p, n, 1, b, x.Matrix(n, opts), df.Matrix(p, opts), rwork))
}
