package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dglmts tests DGGGLM - a subroutine for solving the generalized
// linear model problem.
func dglmts(n, m, p int, a, af, b, bf *mat.Matrix, d, df, x, u, work *mat.Vector, lwork int, rwork *mat.Vector) float64 {
	var anorm, bnorm, dnorm, eps, one, result, unfl, xnorm, ynorm, zero float64
	var err error

	zero = 0.0
	one = 1.0

	eps = golapack.Dlamch(Epsilon)
	unfl = golapack.Dlamch(SafeMinimum)
	anorm = math.Max(golapack.Dlange('1', n, m, a, rwork), unfl)
	bnorm = math.Max(golapack.Dlange('1', n, p, b, rwork), unfl)

	//     Copy the matrices A and B to the arrays AF and BF,
	//     and the vector D the array DF.
	golapack.Dlacpy(Full, n, m, a, af)
	golapack.Dlacpy(Full, n, p, b, bf)
	df.Copy(n, d, 1, 1)

	//     Solve GLM problem
	if _, err = golapack.Dggglm(n, m, p, af, bf, df, x, u, work, lwork); err != nil {
		panic(err)
	}

	//     Test the residual for the solution of LSE
	//
	//                       norm( d - A*x - B*u )
	//       RESULT = -----------------------------------------
	//                (norm(A)+norm(B))*(norm(x)+norm(u))*EPS
	df.Copy(n, d, 1, 1)
	if err = df.Gemv(NoTrans, n, m, -one, a, x, 1, one, 1); err != nil {
		panic(err)
	}

	if err = df.Gemv(NoTrans, n, p, -one, b, u, 1, one, 1); err != nil {
		panic(err)
	}

	dnorm = df.Asum(n, 1)
	xnorm = x.Asum(m, 1) + u.Asum(p, 1)
	ynorm = anorm + bnorm

	if xnorm <= zero {
		result = zero
	} else {
		result = ((dnorm / ynorm) / xnorm) / eps
	}

	return result
}
