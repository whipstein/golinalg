package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zgbt02 computes the residual for a solution of a banded system of
// equations  A*x = b  or  A'*x = b:
//    RESID = norm( B - A*X ) / ( norm(A) * norm(X) * EPS).
// where EPS is the machine precision.
func zgbt02(trans mat.MatTrans, m, n, kl, ku, nrhs int, a, x, b *mat.CMatrix) (resid float64) {
	var cone complex128
	var anorm, bnorm, eps, one, xnorm, zero float64
	var i1, i2, j, kd, n1 int
	var err error

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)
	//     Quick return if N = 0 pr NRHS = 0
	if m <= 0 || n <= 0 || nrhs <= 0 {
		resid = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	kd = ku + 1
	anorm = zero
	for j = 1; j <= n; j++ {
		i1 = max(kd+1-j, 1)
		i2 = min(kd+m-j, kl+kd)
		anorm = math.Max(anorm, goblas.Dzasum(i2-i1+1, a.CVector(i1-1, j-1, 1)))
	}
	if anorm <= zero {
		resid = one / eps
		return
	}

	if trans == Trans || trans == ConjTrans {
		n1 = n
	} else {
		n1 = m
	}

	//     Compute  B - A*X (or  B - A'*X )
	for j = 1; j <= nrhs; j++ {
		if err = goblas.Zgbmv(trans, m, n, kl, ku, -cone, a, x.CVector(0, j-1, 1), cone, b.CVector(0, j-1, 1)); err != nil {
			panic(err)
		}
	}

	//     Compute the maximum over the number of right hand sides of
	//        norm(B - A*X) / ( norm(A) * norm(X) * EPS ).
	resid = zero
	for j = 1; j <= nrhs; j++ {
		bnorm = goblas.Dzasum(n1, b.CVector(0, j-1, 1))
		xnorm = goblas.Dzasum(n1, x.CVector(0, j-1, 1))
		if xnorm <= zero {
			resid = one / eps
		} else {
			resid = math.Max(resid, ((bnorm/anorm)/xnorm)/eps)
		}
	}

	return
}
