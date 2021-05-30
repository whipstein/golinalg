package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
)

// Zsyt03 computes the residual for a complex symmetric matrix times
// its inverse:
//    norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS )
// where EPS is the machine epsilon.
func Zsyt03(uplo byte, n *int, a *mat.CMatrix, lda *int, ainv *mat.CMatrix, ldainv *int, work *mat.CMatrix, ldwork *int, rwork *mat.Vector, rcond, resid *float64) {
	var cone, czero complex128
	var ainvnm, anorm, eps, one, zero float64
	var i, j int

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Quick exit if N = 0
	if (*n) <= 0 {
		(*rcond) = one
		(*resid) = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0 or AINVNM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Zlansy('1', uplo, n, a, lda, rwork)
	ainvnm = golapack.Zlansy('1', uplo, n, ainv, ldainv, rwork)
	if anorm <= zero || ainvnm <= zero {
		(*rcond) = zero
		(*resid) = one / eps
		return
	}
	(*rcond) = (one / anorm) / ainvnm

	//     Expand AINV into a full matrix and call ZSYMM to multiply
	//     AINV on the left by A (store the result in WORK).
	if uplo == 'U' {
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= j-1; i++ {
				ainv.Set(j-1, i-1, ainv.Get(i-1, j-1))
			}
		}
	} else {
		for j = 1; j <= (*n); j++ {
			for i = j + 1; i <= (*n); i++ {
				ainv.Set(j-1, i-1, ainv.Get(i-1, j-1))
			}
		}
	}
	goblas.Zsymm(Left, mat.UploByte(uplo), n, n, toPtrc128(-cone), a, lda, ainv, ldainv, &czero, work, ldwork)

	//     Add the identity matrix to WORK .
	for i = 1; i <= (*n); i++ {
		work.Set(i-1, i-1, work.Get(i-1, i-1)+cone)
	}

	//     Compute norm(I - A*AINV) / (N * norm(A) * norm(AINV) * EPS)
	(*resid) = golapack.Zlange('1', n, n, work, ldwork, rwork)

	(*resid) = (((*resid) * (*rcond)) / eps) / float64(*n)
}
