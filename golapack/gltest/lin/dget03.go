package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dget03 computes the residual for a general matrix times its inverse:
//    norm( I - AINV*A ) / ( N * norm(A) * norm(AINV) * EPS ),
// where EPS is the machine epsilon.
func dget03(n int, a *mat.Matrix, ainv *mat.Matrix, work *mat.Matrix, rwork *mat.Vector) (rcond float64, resid float64, err error) {
	var ainvnm, anorm, eps, one, zero float64
	var i int

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0.
	if n <= 0 {
		rcond = one
		resid = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0 or AINVNM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Dlange('1', n, n, a, rwork)
	ainvnm = golapack.Dlange('1', n, n, ainv, rwork)
	if anorm <= zero || ainvnm <= zero {
		rcond = zero
		resid = one / eps
		return
	}
	rcond = (one / anorm) / ainvnm

	//     Compute I - A * AINV
	err = work.Gemm(NoTrans, NoTrans, n, n, n, -one, ainv, a, zero)
	for i = 1; i <= n; i++ {
		work.Set(i-1, i-1, one+work.Get(i-1, i-1))
	}

	//     Compute norm(I - AINV*A) / (N * norm(A) * norm(AINV) * EPS)
	resid = golapack.Dlange('1', n, n, work, rwork)

	resid = ((resid * rcond) / eps) / float64(n)

	return
}
