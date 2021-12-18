package lin

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dgbt02 computes the residual for a solution of a banded system of
// equations  A*x = b  or  A'*x = b:
//    RESID = norm( B - A*X ) / ( norm(A) * norm(X) * EPS).
// where EPS is the machine precision.
func dgbt02(trans mat.MatTrans, m, n, kl, ku, nrhs int, a *mat.Matrix, x *mat.Matrix, b *mat.Matrix) (resid float64) {
	var anorm, bnorm, eps, one, xnorm, zero float64
	var i1, i2, j, kd, n1 int
	var err error

	zero = 0.0
	one = 1.0

	//     Quick return if N = 0 pr NRHS = 0
	if m <= 0 || n <= 0 || nrhs <= 0 {
		resid = zero
		return resid
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	kd = ku + 1
	anorm = zero
	for j = 1; j <= n; j++ {
		i1 = max(kd+1-j, 1)
		i2 = min(kd+m-j, kl+kd)
		anorm = math.Max(anorm, a.Off(i1-1, j-1).Vector().Asum(i2-i1+1, 1))
	}
	if anorm <= zero {
		resid = one / eps
		return resid
	}

	if trans.IsTrans() {
		n1 = n
	} else {
		n1 = m
	}

	//     Compute  B - A*X (or  B - A'*X )
	for j = 1; j <= nrhs; j++ {
		if err = b.Off(0, j-1).Vector().Gbmv(trans, m, n, kl, ku, -one, a, x.Off(0, j-1).Vector(), 1, one, 1); err != nil {
			panic(err)
		}
	}
	//
	//     Compute the maximum over the number of right hand sides of
	//        norm(B - A*X) / ( norm(A) * norm(X) * EPS ).
	//
	resid = zero
	for j = 1; j <= nrhs; j++ {
		bnorm = b.Off(0, j-1).Vector().Asum(n1, 1)
		xnorm = x.Off(0, j-1).Vector().Asum(n1, 1)
		if xnorm <= zero {
			resid = one / eps
		} else {
			resid = math.Max(resid, ((bnorm/anorm)/xnorm)/eps)
		}
	}

	return resid
}
