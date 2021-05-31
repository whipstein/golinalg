package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zqrt16 computes the residual for a solution of a system of linear
// equations  A*x = b  or  A'*x = b:
//    RESID = norm(B - A*X) / ( maxint(m,n) * norm(A) * norm(X) * EPS ),
// where EPS is the machine epsilon.
func Zqrt16(trans byte, m, n, nrhs *int, a *mat.CMatrix, lda *int, x *mat.CMatrix, ldx *int, b *mat.CMatrix, ldb *int, rwork *mat.Vector, resid *float64) {
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
		anorm = golapack.Zlange('I', m, n, a, lda, rwork)
		n1 = (*n)
		n2 = (*m)
	} else {
		anorm = golapack.Zlange('1', m, n, a, lda, rwork)
		n1 = (*m)
		n2 = (*n)
	}

	eps = golapack.Dlamch(Epsilon)

	//     Compute  B - A*X  (or  B - A'*X ) and store in B.
	goblas.Zgemm(mat.TransByte(trans), NoTrans, &n1, nrhs, &n2, toPtrc128(-cone), a, lda, x, ldx, &cone, b, ldb)

	//     Compute the maximum over the number of right hand sides of
	//        norm(B - A*X) / ( maxint(m,n) * norm(A) * norm(X) * EPS ) .
	(*resid) = zero
	for j = 1; j <= (*nrhs); j++ {
		bnorm = goblas.Dzasum(&n1, b.CVector(0, j-1), func() *int { y := 1; return &y }())
		xnorm = goblas.Dzasum(&n2, x.CVector(0, j-1), func() *int { y := 1; return &y }())
		if anorm == zero && bnorm == zero {
			(*resid) = zero
		} else if anorm <= zero || xnorm <= zero {
			(*resid) = one / eps
		} else {
			(*resid) = maxf64(*resid, ((bnorm/anorm)/xnorm)/(float64(maxint(*m, *n))*eps))
		}
	}
}
