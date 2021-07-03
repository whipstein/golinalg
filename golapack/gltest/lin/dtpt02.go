package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dtpt02 computes the residual for the computed solution to a
// triangular system of linear equations  A*x = b  or  A'*x = b  when
// the triangular matrix A is stored in packed format.  Here A' is the
// transpose of A and x and b are N by NRHS matrices.  The test ratio is
// the maximum over the number of right hand sides of
//    norm(b - op(A)*x) / ( norm(op(A)) * norm(x) * EPS ),
// where op(A) denotes A or A' and EPS is the machine epsilon.
func Dtpt02(uplo, trans, diag byte, n, nrhs *int, ap *mat.Vector, x *mat.Matrix, ldx *int, b *mat.Matrix, ldb *int, work *mat.Vector, resid *float64) {
	var anorm, bnorm, eps, one, xnorm, zero float64
	var j int
	var err error
	_ = err

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0 or NRHS = 0
	if (*n) <= 0 || (*nrhs) <= 0 {
		(*resid) = zero
		return
	}

	//     Compute the 1-norm of A or A'.
	if trans == 'N' {
		anorm = golapack.Dlantp('1', uplo, diag, n, ap, work)
	} else {
		anorm = golapack.Dlantp('I', uplo, diag, n, ap, work)
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	if anorm <= zero {
		(*resid) = one / eps
		return
	}

	//     Compute the maximum over the number of right hand sides of
	//        norm(op(A)*x - b) / ( norm(op(A)) * norm(x) * EPS ).
	(*resid) = zero
	for j = 1; j <= (*nrhs); j++ {
		goblas.Dcopy(*n, x.Vector(0, j-1), 1, work, 1)
		err = goblas.Dtpmv(mat.UploByte(uplo), mat.TransByte(trans), mat.DiagByte(diag), *n, ap, work, 1)
		goblas.Daxpy(*n, -one, b.Vector(0, j-1), 1, work, 1)
		bnorm = goblas.Dasum(*n, work, 1)
		xnorm = goblas.Dasum(*n, x.Vector(0, j-1), 1)
		if xnorm <= zero {
			(*resid) = one / eps
		} else {
			(*resid) = maxf64(*resid, ((bnorm/anorm)/xnorm)/eps)
		}
	}
}
