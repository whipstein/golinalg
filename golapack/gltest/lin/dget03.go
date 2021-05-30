package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
)

// Dget03 computes the residual for a general matrix times its inverse:
//    norm( I - AINV*A ) / ( N * norm(A) * norm(AINV) * EPS ),
// where EPS is the machine epsilon.
func Dget03(n *int, a *mat.Matrix, lda *int, ainv *mat.Matrix, ldainv *int, work *mat.Matrix, ldwork *int, rwork *mat.Vector, rcond *float64, resid *float64) {
	var ainvnm, anorm, eps, one, zero float64
	var i int

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0.
	if (*n) <= 0 {
		(*rcond) = one
		(*resid) = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0 or AINVNM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Dlange('1', n, n, a, lda, rwork)
	ainvnm = golapack.Dlange('1', n, n, ainv, ldainv, rwork)
	if anorm <= zero || ainvnm <= zero {
		(*rcond) = zero
		(*resid) = one / eps
		return
	}
	(*rcond) = (one / anorm) / ainvnm

	//     Compute I - A * AINV
	goblas.Dgemm(mat.NoTrans, mat.NoTrans, n, n, n, toPtrf64(-one), ainv, ldainv, a, lda, &zero, work, ldwork)
	for i = 1; i <= (*n); i++ {
		work.Set(i-1, i-1, one+work.Get(i-1, i-1))
	}

	//     Compute norm(I - AINV*A) / (N * norm(A) * norm(AINV) * EPS)
	(*resid) = golapack.Dlange('1', n, n, work, ldwork, rwork)

	(*resid) = (((*resid) * (*rcond)) / eps) / float64(*n)
}
