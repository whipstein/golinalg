package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zptt02 computes the residual for the solution to a symmetric
// tridiagonal system of equations:
//    RESID = norm(B - A*X) / (norm(A) * norm(X) * EPS),
// where EPS is the machine epsilon.
func zptt02(uplo mat.MatUplo, n, nrhs int, d *mat.Vector, e *mat.CVector, x, b *mat.CMatrix) (resid float64) {
	var anorm, bnorm, eps, one, xnorm, zero float64
	var j int

	one = 1.0
	zero = 0.0

	//     Quick return if possible
	if n <= 0 {
		resid = zero
		return
	}

	//     Compute the 1-norm of the tridiagonal matrix A.
	anorm = golapack.Zlanht('1', n, d, e)

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	if anorm <= zero {
		resid = one / eps
		return
	}

	//     Compute B - A*X.
	zlaptm(uplo, n, nrhs, -one, d, e, x, one, b)

	//     Compute the maximum over the number of right hand sides of
	//        norm(B - A*X) / ( norm(A) * norm(X) * EPS ).
	resid = zero
	for j = 1; j <= nrhs; j++ {
		bnorm = goblas.Dzasum(n, b.CVector(0, j-1, 1))
		xnorm = goblas.Dzasum(n, x.CVector(0, j-1, 1))
		if xnorm <= zero {
			resid = one / eps
		} else {
			resid = math.Max(resid, ((bnorm/anorm)/xnorm)/eps)
		}
	}

	return
}
