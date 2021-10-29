package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zpbt02 computes the residual for a solution of a Hermitian banded
// system of equations  A*x = b:
//    RESID = norm( B - A*X ) / ( norm(A) * norm(X) * EPS)
// where EPS is the machine precision.
func zpbt02(uplo mat.MatUplo, n, kd, nrhs int, a, x, b *mat.CMatrix, rwork *mat.Vector) (resid float64) {
	var cone complex128
	var anorm, bnorm, eps, one, xnorm, zero float64
	var j int
	var err error

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)

	//     Quick exit if N = 0 or NRHS = 0.
	if n <= 0 || nrhs <= 0 {
		resid = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Zlanhb('1', uplo, n, kd, a, rwork)
	if anorm <= zero {
		resid = one / eps
		return
	}

	//     Compute  B - A*X
	for j = 1; j <= nrhs; j++ {
		if err = goblas.Zhbmv(uplo, n, kd, -cone, a, x.CVector(0, j-1, 1), cone, b.CVector(0, j-1, 1)); err != nil {
			panic(err)
		}
	}

	//     Compute the maximum over the number of right hand sides of
	//          norm( B - A*X ) / ( norm(A) * norm(X) * EPS )
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
