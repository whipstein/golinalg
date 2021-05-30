package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
)

// Zptt02 computes the residual for the solution to a symmetric
// tridiagonal system of equations:
//    RESID = norm(B - A*X) / (norm(A) * norm(X) * EPS),
// where EPS is the machine epsilon.
func Zptt02(uplo byte, n, nrhs *int, d *mat.Vector, e *mat.CVector, x *mat.CMatrix, ldx *int, b *mat.CMatrix, ldb *int, resid *float64) {
	var anorm, bnorm, eps, one, xnorm, zero float64
	var j int

	one = 1.0
	zero = 0.0

	//     Quick return if possible
	if (*n) <= 0 {
		(*resid) = zero
		return
	}

	//     Compute the 1-norm of the tridiagonal matrix A.
	anorm = golapack.Zlanht('1', n, d, e)

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	if anorm <= zero {
		(*resid) = one / eps
		return
	}

	//     Compute B - A*X.
	Zlaptm(uplo, n, nrhs, toPtrf64(-one), d, e, x, ldx, &one, b, ldb)

	//     Compute the maximum over the number of right hand sides of
	//        norm(B - A*X) / ( norm(A) * norm(X) * EPS ).
	(*resid) = zero
	for j = 1; j <= (*nrhs); j++ {
		bnorm = goblas.Dzasum(n, b.CVector(0, j-1), func() *int { y := 1; return &y }())
		xnorm = goblas.Dzasum(n, x.CVector(0, j-1), func() *int { y := 1; return &y }())
		if xnorm <= zero {
			(*resid) = one / eps
		} else {
			(*resid) = maxf64(*resid, ((bnorm/anorm)/xnorm)/eps)
		}
	}
}
