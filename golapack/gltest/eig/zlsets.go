package eig

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zlsets tests ZGGLSE - a subroutine for solving linear equality
// constrained least square problem (LSE).
func zlsets(m, p, n int, a, af, b, bf *mat.CMatrix, c, cf, d, df, x, work *mat.CVector, lwork int, rwork, result *mat.Vector) {
	var err error

	//     Copy the matrices A and B to the arrays AF and BF,
	//     and the vectors C and D to the arrays CF and DF,
	golapack.Zlacpy(Full, m, n, a, af)
	golapack.Zlacpy(Full, p, n, b, bf)
	cf.Copy(m, c, 1, 1)
	df.Copy(p, d, 1, 1)

	//     Solve LSE problem
	if _, err = golapack.Zgglse(m, n, p, af, bf, cf, df, x, work, lwork); err != nil {
		panic(err)
	}

	//     Test the residual for the solution of LSE
	//
	//     Compute RESULT(1) = norm( A*x - c ) / norm(A)*norm(X)*EPS
	cf.Copy(m, c, 1, 1)
	df.Copy(p, d, 1, 1)
	result.Set(0, zget02(NoTrans, m, n, 1, a, x.CMatrix(n, opts), cf.CMatrix(m, opts), rwork))

	//     Compute result(2) = norm( B*x - d ) / norm(B)*norm(X)*EPS
	result.Set(1, zget02(NoTrans, p, n, 1, b, x.CMatrix(n, opts), df.CMatrix(p, opts), rwork))
}
