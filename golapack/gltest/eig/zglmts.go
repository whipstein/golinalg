package eig

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zglmts tests ZGGGLM - a subroutine for solving the generalized
// linear model problem.
func Zglmts(n, m, p *int, a, af *mat.CMatrix, lda *int, b, bf *mat.CMatrix, ldb *int, d, df, x, u, work *mat.CVector, lwork *int, rwork *mat.Vector, result *float64) {
	var cone complex128
	var anorm, bnorm, dnorm, eps, unfl, xnorm, ynorm, zero float64
	var info int
	var err error
	_ = err

	zero = 0.0
	cone = 1.0

	eps = golapack.Dlamch(Epsilon)
	unfl = golapack.Dlamch(SafeMinimum)
	anorm = maxf64(golapack.Zlange('1', n, m, a, lda, rwork), unfl)
	bnorm = maxf64(golapack.Zlange('1', n, p, b, ldb, rwork), unfl)

	//     Copy the matrices A and B to the arrays AF and BF,
	//     and the vector D the array DF.
	golapack.Zlacpy('F', n, m, a, lda, af, lda)
	golapack.Zlacpy('F', n, p, b, ldb, bf, ldb)
	goblas.Zcopy(*n, d, 1, df, 1)

	//     Solve GLM problem
	golapack.Zggglm(n, m, p, af, lda, bf, ldb, df, x, u, work, lwork, &info)

	//     Test the residual for the solution of LSE
	//
	//                       norm( d - A*x - B*u )
	//       RESULT = -----------------------------------------
	//                (norm(A)+norm(B))*(norm(x)+norm(u))*EPS
	goblas.Zcopy(*n, d, 1, df, 1)
	goblas.Zgemv(NoTrans, *n, *m, -cone, a, *lda, x, 1, cone, df, 1)

	goblas.Zgemv(NoTrans, *n, *p, -cone, b, *ldb, u, 1, cone, df, 1)

	dnorm = goblas.Dzasum(*n, df, 1)
	xnorm = goblas.Dzasum(*m, x, 1) + goblas.Dzasum(*p, u, 1)
	ynorm = anorm + bnorm

	if xnorm <= zero {
		(*result) = zero
	} else {
		(*result) = ((dnorm / ynorm) / xnorm) / eps
	}
}
