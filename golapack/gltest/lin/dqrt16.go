package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
)

// Dqrt16 computes the residual for a solution of a system of linear
// equations  A*x = b  or  A'*x = b:
//    RESID = norm(B - A*X) / ( max(m,n) * norm(A) * norm(X) * EPS ),
// where EPS is the machine epsilon.
func Dqrt16(trans byte, m, n, nrhs *int, a *mat.Matrix, lda *int, x *mat.Matrix, ldx *int, b *mat.Matrix, ldb *int, rwork *mat.Vector, resid *float64) {
	var anorm, bnorm, eps, one, xnorm, zero float64
	var j, n1, n2 int

	zero = 0.0
	one = 1.0

	//     Quick exit if M = 0 or N = 0 or NRHS = 0
	if (*m) <= 0 || (*n) <= 0 || (*nrhs) == 0 {
		(*resid) = zero
		return
	}

	if trans == 'T' || trans == 'C' {
		anorm = golapack.Dlange('I', m, n, a, lda, rwork)
		n1 = (*n)
		n2 = (*m)
	} else {
		anorm = golapack.Dlange('1', m, n, a, lda, rwork)
		n1 = (*m)
		n2 = (*n)
	}

	eps = golapack.Dlamch(Epsilon)

	//     Compute  B - A*X  (or  B - A'*X ) and store in B.
	goblas.Dgemm(mat.TransByte(trans), NoTrans, &n1, nrhs, &n2, toPtrf64(-one), a, lda, x, ldx, &one, b, ldb)

	//     Compute the maximum over the number of right hand sides of
	//        norm(B - A*X) / ( max(m,n) * norm(A) * norm(X) * EPS ) .
	(*resid) = zero
	for j = 1; j <= (*nrhs); j++ {
		bnorm = goblas.Dasum(&n1, b.Vector(0, j-1), toPtr(1))
		xnorm = goblas.Dasum(&n2, x.Vector(0, j-1), toPtr(1))
		if anorm == zero && bnorm == zero {
			(*resid) = zero
		} else if anorm <= zero || xnorm <= zero {
			(*resid) = one / eps
		} else {
			(*resid) = maxf64(*resid, ((bnorm/anorm)/xnorm)/(float64(maxint(*m, *n))*eps))
		}
	}
}