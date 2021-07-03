package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zppt02 computes the residual in the solution of a Hermitian system
// of linear equations  A*x = b  when packed storage is used for the
// coefficient matrix.  The ratio computed is
//
//    RESID = norm(B - A*X) / ( norm(A) * norm(X) * EPS),
//
// where EPS is the machine precision.
func Zppt02(uplo byte, n, nrhs *int, a *mat.CVector, x *mat.CMatrix, ldx *int, b *mat.CMatrix, ldb *int, rwork *mat.Vector, resid *float64) {
	var cone complex128
	var anorm, bnorm, eps, one, xnorm, zero float64
	var j int
	var err error
	_ = err

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
	anorm = golapack.Zlanhp('1', uplo, n, a, rwork)
	if anorm <= zero {
		(*resid) = one / eps
		return
	}

	//     Compute  B - A*X  for the matrix of right hand sides B.
	for j = 1; j <= (*nrhs); j++ {
		err = goblas.Zhpmv(mat.UploByte(uplo), *n, -cone, a, x.CVector(0, j-1), 1, cone, b.CVector(0, j-1), 1)
	}

	//     Compute the maximum over the number of right hand sides of
	//        norm( B - A*X ) / ( norm(A) * norm(X) * EPS ) .
	(*resid) = zero
	for j = 1; j <= (*nrhs); j++ {
		bnorm = goblas.Dzasum(*n, b.CVector(0, j-1), 1)
		xnorm = goblas.Dzasum(*n, x.CVector(0, j-1), 1)
		if xnorm <= zero {
			(*resid) = one / eps
		} else {
			(*resid) = maxf64(*resid, ((bnorm/anorm)/xnorm)/eps)
		}
	}
}
