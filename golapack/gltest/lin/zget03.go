package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zget03 computes the residual for a general matrix times its inverse:
//    norm( I - AINV*A ) / ( N * norm(A) * norm(AINV) * EPS ),
// where EPS is the machine epsilon.
func zget03(n int, a, ainv, work *mat.CMatrix, rwork *mat.Vector) (rcond, resid float64) {
	var cone, czero complex128
	var ainvnm, anorm, eps, one, zero float64
	var i int
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
	anorm = golapack.Zlange('1', n, n, a, rwork)
	ainvnm = golapack.Zlange('1', n, n, ainv, rwork)
	if anorm <= zero || ainvnm <= zero {
		rcond = zero
		resid = one / eps
		return
	}
	rcond = (one / anorm) / ainvnm

	//     Compute I - A * AINV
	if err = work.Gemm(NoTrans, NoTrans, n, n, n, -cone, ainv, a, czero); err != nil {
		panic(err)
	}
	for i = 1; i <= n; i++ {
		work.Set(i-1, i-1, cone+work.Get(i-1, i-1))
	}

	//     Compute norm(I - AINV*A) / (N * norm(A) * norm(AINV) * EPS)
	resid = golapack.Zlange('1', n, n, work, rwork)

	resid = ((resid * rcond) / eps) / float64(n)

	return
}
