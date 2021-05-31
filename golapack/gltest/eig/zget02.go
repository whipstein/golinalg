package eig

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zget02 computes the residual for a solution of a system of linear
// equations  A*x = b  or  A'*x = b:
//    RESID = norm(B - A*X) / ( norm(A) * norm(X) * EPS ),
// where EPS is the machine epsilon.
func Zget02(trans byte, m, n, nrhs *int, a *mat.CMatrix, lda *int, x *mat.CMatrix, ldx *int, b *mat.CMatrix, ldb *int, rwork *mat.Vector, resid *float64) {
	var cone complex128
	var anorm, bnorm, eps, one, xnorm, zero float64
	var j, n1, n2 int

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)

	//     Quick exit if M = 0 or N = 0 or NRHS = 0
	if (*m) <= 0 || (*n) <= 0 || (*nrhs) == 0 {
		(*resid) = zero
		return
	}

	if trans == 'T' || trans == 'C' {
		n1 = (*n)
		n2 = (*m)
	} else {
		n1 = (*m)
		n2 = (*n)
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Zlange('1', m, n, a, lda, rwork)
	if anorm <= zero {
		(*resid) = one / eps
		return
	}

	//     Compute  B - A*X  (or  B - A'*X ) and store in B.
	goblas.Zgemm(mat.TransByte(trans), NoTrans, &n1, nrhs, &n2, toPtrc128(-cone), a, lda, x, ldx, &cone, b, ldb)

	//     Compute the maximum over the number of right hand sides of
	//        norm(B - A*X) / ( norm(A) * norm(X) * EPS ) .
	(*resid) = zero
	for j = 1; j <= (*nrhs); j++ {
		bnorm = goblas.Dzasum(&n1, b.CVector(0, j-1), func() *int { y := 1; return &y }())
		xnorm = goblas.Dzasum(&n2, x.CVector(0, j-1), func() *int { y := 1; return &y }())
		if xnorm <= zero {
			(*resid) = one / eps
		} else {
			(*resid) = maxf64(*resid, ((bnorm/anorm)/xnorm)/eps)
		}
	}
}
