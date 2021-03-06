package lin

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dpot02 computes the residual for the solution of a symmetric system
// of linear equations  A*x = b:
//
//    RESID = norm(B - A*X) / ( norm(A) * norm(X) * EPS ),
//
// where EPS is the machine epsilon.
func dpot02(uplo mat.MatUplo, n, nrhs int, a, x, b *mat.Matrix, rwork *mat.Vector) (resid float64) {
	var anorm, bnorm, eps, one, xnorm, zero float64
	var j int
	var err error

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0 or NRHS = 0.
	if n <= 0 || nrhs <= 0 {
		resid = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Dlansy('1', uplo, n, a, rwork)
	if anorm <= zero {
		resid = one / eps
		return
	}

	//     Compute  B - A*X
	if err = b.Symm(Left, uplo, n, nrhs, -one, a, x, one); err != nil {
		panic(err)
	}

	//     Compute the maximum over the number of right hand sides of
	//        norm( B - A*X ) / ( norm(A) * norm(X) * EPS ) .
	resid = zero
	for j = 1; j <= nrhs; j++ {
		bnorm = b.Off(0, j-1).Vector().Asum(n, 1)
		xnorm = x.Off(0, j-1).Vector().Asum(n, 1)
		if xnorm <= zero {
			resid = one / eps
		} else {
			resid = math.Max(resid, ((bnorm/anorm)/xnorm)/eps)
		}
	}

	return
}
