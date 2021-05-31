package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dgbt02 computes the residual for a solution of a banded system of
// equations  A*x = b  or  A'*x = b:
//    RESID = norm( B - A*X ) / ( norm(A) * norm(X) * EPS).
// where EPS is the machine precision.
func Dgbt02(trans byte, m, n, kl, ku, nrhs *int, a *mat.Matrix, lda *int, x *mat.Matrix, ldx *int, b *mat.Matrix, ldb *int, resid *float64) {
	var anorm, bnorm, eps, one, xnorm, zero float64
	var i1, i2, j, kd, n1 int

	zero = 0.0
	one = 1.0

	//     Quick return if N = 0 pr NRHS = 0
	if (*m) <= 0 || (*n) <= 0 || (*nrhs) <= 0 {
		(*resid) = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	kd = (*ku) + 1
	anorm = zero
	for j = 1; j <= (*n); j++ {
		i1 = maxint(kd+1-j, 1)
		i2 = minint(kd+(*m)-j, (*kl)+kd)
		anorm = maxf64(anorm, goblas.Dasum(toPtr(i2-i1+1), a.Vector(i1-1, j-1), toPtr(1)))
	}
	if anorm <= zero {
		(*resid) = one / eps
		return
	}

	if trans == 'T' || trans == 'C' {
		n1 = (*n)
	} else {
		n1 = (*m)
	}

	//     Compute  B - A*X (or  B - A'*X )
	for j = 1; j <= (*nrhs); j++ {
		goblas.Dgbmv(mat.TransByte(trans), m, n, kl, ku, toPtrf64(-one), a, lda, x.Vector(0, j-1), toPtr(1), &one, b.Vector(0, j-1), toPtr(1))
	}
	//
	//     Compute the maximum over the number of right hand sides of
	//        norm(B - A*X) / ( norm(A) * norm(X) * EPS ).
	//
	(*resid) = zero
	for j = 1; j <= (*nrhs); j++ {
		bnorm = goblas.Dasum(&n1, b.Vector(0, j-1), toPtr(1))
		xnorm = goblas.Dasum(&n1, x.Vector(0, j-1), toPtr(1))
		if xnorm <= zero {
			(*resid) = one / eps
		} else {
			(*resid) = maxf64(*resid, ((bnorm/anorm)/xnorm)/eps)
		}
	}
}
