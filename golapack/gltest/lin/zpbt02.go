package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zpbt02 computes the residual for a solution of a Hermitian banded
// system of equations  A*x = b:
//    RESID = norm( B - A*X ) / ( norm(A) * norm(X) * EPS)
// where EPS is the machine precision.
func Zpbt02(uplo byte, n, kd, nrhs *int, a *mat.CMatrix, lda *int, x *mat.CMatrix, ldx *int, b *mat.CMatrix, ldb *int, rwork *mat.Vector, resid *float64) {
	var cone complex128
	var anorm, bnorm, eps, one, xnorm, zero float64
	var j int

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)

	//     Quick exit if N = 0 or NRHS = 0.
	if (*n) <= 0 || (*nrhs) <= 0 {
		(*resid) = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Zlanhb('1', uplo, n, kd, a, lda, rwork)
	if anorm <= zero {
		(*resid) = one / eps
		return
	}

	//     Compute  B - A*X
	for j = 1; j <= (*nrhs); j++ {
		goblas.Zhbmv(mat.UploByte(uplo), n, kd, toPtrc128(-cone), a, lda, x.CVector(0, j-1), func() *int { y := 1; return &y }(), &cone, b.CVector(0, j-1), func() *int { y := 1; return &y }())
	}

	//     Compute the maximum over the number of right hand sides of
	//          norm( B - A*X ) / ( norm(A) * norm(X) * EPS )
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
