package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// ztpt02 computes the residual for the computed solution to a
// triangular system of linear equations  A*x = b,  A**T *x = b,  or
// A**H *x = b, when the triangular matrix A is stored in packed format.
// Here A**T denotes the transpose of A, A**H denotes the conjugate
// transpose of A, and x and b are N by NRHS matrices.  The test ratio
// is the maximum over the number of right hand sides of
// the maximum over the number of right hand sides of
//    norm(b - op(A)*x) / ( norm(op(A)) * norm(x) * EPS ),
// where op(A) denotes A, A**T, or A**H, and EPS is the machine epsilon.
func ztpt02(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, nrhs int, ap *mat.CVector, x, b *mat.CMatrix, work *mat.CVector, rwork *mat.Vector) (resid float64) {
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

	//     Compute the 1-norm of A or A**H.
	if trans == NoTrans {
		anorm = golapack.Zlantp('1', uplo, diag, n, ap, rwork)
	} else {
		anorm = golapack.Zlantp('I', uplo, diag, n, ap, rwork)
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	if anorm <= zero {
		resid = one / eps
		return
	}

	//     Compute the maximum over the number of right hand sides of
	//        norm(op(A)*x - b) / ( norm(op(A)) * norm(x) * EPS ).
	resid = zero
	for j = 1; j <= nrhs; j++ {
		goblas.Zcopy(n, x.CVector(0, j-1, 1), work.Off(0, 1))
		if err = goblas.Ztpmv(uplo, trans, diag, n, ap, work.Off(0, 1)); err != nil {
			panic(err)
		}
		goblas.Zaxpy(n, complex(-one, 0), b.CVector(0, j-1, 1), work.Off(0, 1))
		bnorm = goblas.Dzasum(n, work.Off(0, 1))
		xnorm = goblas.Dzasum(n, x.CVector(0, j-1, 1))
		if xnorm <= zero {
			resid = one / eps
		} else {
			resid = math.Max(resid, ((bnorm/anorm)/xnorm)/eps)
		}
	}

	return
}
