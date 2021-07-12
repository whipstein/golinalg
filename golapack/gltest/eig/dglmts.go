package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dglmts tests DGGGLM - a subroutine for solving the generalized
// linear model problem.
func Dglmts(n, m, p *int, a, af *mat.Matrix, lda *int, b, bf *mat.Matrix, ldb *int, d, df, x, u, work *mat.Vector, lwork *int, rwork *mat.Vector, result *float64) {
	var anorm, bnorm, dnorm, eps, one, unfl, xnorm, ynorm, zero float64
	var info int
	var err error
	_ = err

	zero = 0.0
	one = 1.0

	eps = golapack.Dlamch(Epsilon)
	unfl = golapack.Dlamch(SafeMinimum)
	anorm = math.Max(golapack.Dlange('1', n, m, a, lda, rwork), unfl)
	bnorm = math.Max(golapack.Dlange('1', n, p, b, ldb, rwork), unfl)

	//     Copy the matrices A and B to the arrays AF and BF,
	//     and the vector D the array DF.
	golapack.Dlacpy('F', n, m, a, lda, af, lda)
	golapack.Dlacpy('F', n, p, b, ldb, bf, ldb)
	goblas.Dcopy(*n, d.Off(0, 1), df.Off(0, 1))

	//     Solve GLM problem
	golapack.Dggglm(n, m, p, af, lda, bf, ldb, df, x, u, work, lwork, &info)

	//     Test the residual for the solution of LSE
	//
	//                       norm( d - A*x - B*u )
	//       RESULT = -----------------------------------------
	//                (norm(A)+norm(B))*(norm(x)+norm(u))*EPS
	goblas.Dcopy(*n, d.Off(0, 1), df.Off(0, 1))
	err = goblas.Dgemv(NoTrans, *n, *m, -one, a, x.Off(0, 1), one, df.Off(0, 1))

	err = goblas.Dgemv(NoTrans, *n, *p, -one, b, u.Off(0, 1), one, df.Off(0, 1))

	dnorm = goblas.Dasum(*n, df.Off(0, 1))
	xnorm = goblas.Dasum(*m, x.Off(0, 1)) + goblas.Dasum(*p, u.Off(0, 1))
	ynorm = anorm + bnorm

	if xnorm <= zero {
		(*result) = zero
	} else {
		(*result) = ((dnorm / ynorm) / xnorm) / eps
	}
}
