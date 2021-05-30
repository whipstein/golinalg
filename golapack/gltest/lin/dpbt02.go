package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
)

// Dpbt02 computes the residual for a solution of a symmetric banded
// system of equations  A*x = b:
//    RESID = norm( B - A*X ) / ( norm(A) * norm(X) * EPS)
// where EPS is the machine precision.
func Dpbt02(uplo byte, n, kd, nrhs *int, a *mat.Matrix, lda *int, x *mat.Matrix, ldx *int, b *mat.Matrix, ldb *int, rwork *mat.Vector, resid *float64) {
	var anorm, bnorm, eps, one, xnorm, zero float64
	var j int

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0 or NRHS = 0.
	if (*n) <= 0 || (*nrhs) <= 0 {
		(*resid) = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Dlansb('1', uplo, n, kd, a, lda, rwork)
	if anorm <= zero {
		(*resid) = one / eps
		return
	}

	//     Compute  B - A*X
	for j = 1; j <= (*nrhs); j++ {
		goblas.Dsbmv(mat.UploByte(uplo), n, kd, toPtrf64(-one), a, lda, x.Vector(0, j-1), toPtr(1), &one, b.Vector(0, j-1), toPtr(1))
	}

	//     Compute the maximum over the number of right hand sides of
	//          norm( B - A*X ) / ( norm(A) * norm(X) * EPS )
	(*resid) = zero
	for j = 1; j <= (*nrhs); j++ {
		bnorm = goblas.Dasum(n, b.Vector(0, j-1), toPtr(1))
		xnorm = goblas.Dasum(n, x.Vector(0, j-1), toPtr(1))
		if xnorm <= zero {
			(*resid) = one / eps
		} else {
			(*resid) = maxf64(*resid, ((bnorm/anorm)/xnorm)/eps)
		}
	}
}
