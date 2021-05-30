package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
)

// Ztrt01 computes the residual for a triangular matrix A times its
// inverse:
//    RESID = norm( A*AINV - I ) / ( N * norm(A) * norm(AINV) * EPS ),
// where EPS is the machine epsilon.
func Ztrt01(uplo, diag byte, n *int, a *mat.CMatrix, lda *int, ainv *mat.CMatrix, ldainv *int, rcond *float64, rwork *mat.Vector, resid *float64) {
	var ainvnm, anorm, eps, one, zero float64
	var j int

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0
	if (*n) <= 0 {
		(*rcond) = one
		(*resid) = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0 or AINVNM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Zlantr('1', uplo, diag, n, n, a, lda, rwork)
	ainvnm = golapack.Zlantr('1', uplo, diag, n, n, ainv, ldainv, rwork)
	if anorm <= zero || ainvnm <= zero {
		(*rcond) = zero
		(*resid) = one / eps
		return
	}
	(*rcond) = (one / anorm) / ainvnm

	//     Set the diagonal of AINV to 1 if AINV has unit diagonal.
	if diag == 'U' {
		for j = 1; j <= (*n); j++ {
			ainv.SetRe(j-1, j-1, one)
		}
	}

	//     Compute A * AINV, overwriting AINV.
	if uplo == 'U' {
		for j = 1; j <= (*n); j++ {
			goblas.Ztrmv(Upper, NoTrans, mat.DiagByte(diag), &j, a, lda, ainv.CVector(0, j-1), func() *int { y := 1; return &y }())
		}
	} else {
		for j = 1; j <= (*n); j++ {
			goblas.Ztrmv(Lower, NoTrans, mat.DiagByte(diag), toPtr((*n)-j+1), a.Off(j-1, j-1), lda, ainv.CVector(j-1, j-1), func() *int { y := 1; return &y }())
		}
	}

	//     Subtract 1 from each diagonal element to form A*AINV - I.
	for j = 1; j <= (*n); j++ {
		ainv.Set(j-1, j-1, ainv.Get(j-1, j-1)-complex(one, 0))
	}

	//     Compute norm(A*AINV - I) / (N * norm(A) * norm(AINV) * EPS)
	(*resid) = golapack.Zlantr('1', uplo, 'N', n, n, ainv, ldainv, rwork)

	(*resid) = (((*resid) * (*rcond)) / float64(*n)) / eps
}
