package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
)

// Zgbt02 computes the residual for a solution of a banded system of
// equations  A*x = b  or  A'*x = b:
//    RESID = norm( B - A*X ) / ( norm(A) * norm(X) * EPS).
// where EPS is the machine precision.
func Zgbt02(trans byte, m, n, kl, ku, nrhs *int, a *mat.CMatrix, lda *int, x *mat.CMatrix, ldx *int, b *mat.CMatrix, ldb *int, resid *float64) {
	var cone complex128
	var anorm, bnorm, eps, one, xnorm, zero float64
	var i1, i2, j, kd, n1 int

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)
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
		anorm = maxf64(anorm, goblas.Dzasum(toPtr(i2-i1+1), a.CVector(i1-1, j-1), func() *int { y := 1; return &y }()))
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
		goblas.Zgbmv(mat.TransByte(trans), m, n, kl, ku, toPtrc128(-cone), a, lda, x.CVector(0, j-1), func() *int { y := 1; return &y }(), &cone, b.CVector(0, j-1), func() *int { y := 1; return &y }())
	}

	//     Compute the maximum over the number of right hand sides of
	//        norm(B - A*X) / ( norm(A) * norm(X) * EPS ).
	(*resid) = zero
	for j = 1; j <= (*nrhs); j++ {
		bnorm = goblas.Dzasum(&n1, b.CVector(0, j-1), func() *int { y := 1; return &y }())
		xnorm = goblas.Dzasum(&n1, x.CVector(0, j-1), func() *int { y := 1; return &y }())
		if xnorm <= zero {
			(*resid) = one / eps
		} else {
			(*resid) = maxf64(*resid, ((bnorm/anorm)/xnorm)/eps)
		}
	}
}
