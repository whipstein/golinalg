package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dtbt02 computes the residual for the computed solution to a
// triangular system of linear equations  A*x = b  or  A' *x = b when
// A is a triangular band matrix.  Here A' is the transpose of A and
// x and b are N by NRHS matrices.  The test ratio is the maximum over
// the number of right hand sides of
//    norm(b - op(A)*x) / ( norm(op(A)) * norm(x) * EPS ),
// where op(A) denotes A or A' and EPS is the machine epsilon.
func Dtbt02(uplo, trans, diag byte, n, kd, nrhs *int, ab *mat.Matrix, ldab *int, x *mat.Matrix, ldx *int, b *mat.Matrix, ldb *int, work *mat.Vector, resid *float64) {
	var anorm, bnorm, eps, one, xnorm, zero float64
	var j int

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0 or NRHS = 0
	if (*n) <= 0 || (*nrhs) <= 0 {
		(*resid) = zero
		return
	}

	//     Compute the 1-norm of A or A'.
	if trans == 'N' {
		anorm = golapack.Dlantb('1', uplo, diag, n, kd, ab, ldab, work)
	} else {
		anorm = golapack.Dlantb('I', uplo, diag, n, kd, ab, ldab, work)
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
		goblas.Dcopy(n, x.Vector(0, j-1), toPtr(1), work, toPtr(1))
		goblas.Dtbmv(mat.UploByte(uplo), mat.TransByte(trans), mat.DiagByte(diag), n, kd, ab, ldab, work, toPtr(1))
		goblas.Daxpy(n, toPtrf64(-one), b.Vector(0, j-1), toPtr(1), work, toPtr(1))
		bnorm = goblas.Dasum(n, work, toPtr(1))
		xnorm = goblas.Dasum(n, x.Vector(0, j-1), toPtr(1))
		if xnorm <= zero {
			(*resid) = one / eps
		} else {
			(*resid) = maxf64(*resid, ((bnorm/anorm)/xnorm)/eps)
		}
	}
}
