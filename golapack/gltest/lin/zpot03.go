package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zpot03 computes the residual for a Hermitian matrix times its
// inverse:
//    norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS ),
// where EPS is the machine epsilon.
func zpot03(uplo mat.MatUplo, n int, a, ainv, work *mat.CMatrix, rwork *mat.Vector) (rcond, resid float64) {
	var cone, czero complex128
	var ainvnm, anorm, eps, one, zero float64
	var i, j int
	var err error

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Quick exit if N = 0.
	if n <= 0 {
		rcond = one
		resid = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0 or AINVNM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Zlanhe('1', uplo, n, a, rwork)
	ainvnm = golapack.Zlanhe('1', uplo, n, ainv, rwork)
	if anorm <= zero || ainvnm <= zero {
		rcond = zero
		resid = one / eps
		return
	}
	rcond = (one / anorm) / ainvnm

	//     Expand AINV into a full matrix and call ZHEMM to multiply
	//     AINV on the left by A.
	if uplo == Upper {
		for j = 1; j <= n; j++ {
			for i = 1; i <= j-1; i++ {
				ainv.Set(j-1, i-1, ainv.GetConj(i-1, j-1))
			}
		}
	} else {
		for j = 1; j <= n; j++ {
			for i = j + 1; i <= n; i++ {
				ainv.Set(j-1, i-1, ainv.GetConj(i-1, j-1))
			}
		}
	}
	if err = work.Hemm(Left, uplo, n, n, -cone, a, ainv, czero); err != nil {
		panic(err)
	}

	//     Add the identity matrix to WORK .
	for i = 1; i <= n; i++ {
		work.Set(i-1, i-1, work.Get(i-1, i-1)+cone)
	}

	//     Compute norm(I - A*AINV) / (N * norm(A) * norm(AINV) * EPS)
	resid = golapack.Zlange('1', n, n, work, rwork)

	resid = ((resid * rcond) / eps) / float64(n)

	return
}
