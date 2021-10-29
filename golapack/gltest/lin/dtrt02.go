package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dtrt02 computes the residual for the computed solution to a
// triangular system of linear equations  A*x = b  or  A'*x = b.
// Here A is a triangular matrix, A' is the transpose of A, and x and b
// are N by NRHS matrices.  The test ratio is the maximum over the
// number of right hand sides of
//    norm(b - op(A)*x) / ( norm(op(A)) * norm(x) * EPS ),
// where op(A) denotes A or A' and EPS is the machine epsilon.
func dtrt02(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, nrhs int, a, x, b *mat.Matrix, work *mat.Vector) (resid float64) {
	var anorm, bnorm, eps, one, xnorm, zero float64
	var j int
	var err error

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0 or NRHS = 0
	if n <= 0 || nrhs <= 0 {
		resid = zero
		return
	}

	//     Compute the 1-norm of A or A'.
	if trans == NoTrans {
		anorm = golapack.Dlantr('1', uplo, diag, n, n, a, work)
	} else {
		anorm = golapack.Dlantr('I', uplo, diag, n, n, a, work)
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	if anorm <= zero {
		resid = one / eps
		return
	}

	//     Compute the maximum over the number of right hand sides of
	//        norm(op(A)*x - b) / ( norm(op(A)) * norm(x) * EPS )
	resid = zero
	for j = 1; j <= nrhs; j++ {
		goblas.Dcopy(n, x.Vector(0, j-1, 1), work.Off(0, 1))
		if err = goblas.Dtrmv(uplo, trans, diag, n, a, work.Off(0, 1)); err != nil {
			panic(err)
		}
		goblas.Daxpy(n, -one, b.Vector(0, j-1, 1), work.Off(0, 1))
		bnorm = goblas.Dasum(n, work.Off(0, 1))
		xnorm = goblas.Dasum(n, x.Vector(0, j-1, 1))
		if xnorm <= zero {
			resid = one / eps
		} else {
			resid = math.Max(resid, ((bnorm/anorm)/xnorm)/eps)
		}
	}

	return
}
