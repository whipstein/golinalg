package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zget02 computes the residual for a solution of a system of linear
// equations  A*x = b  or  A'*x = b:
//    RESID = norm(B - A*X) / ( norm(A) * norm(X) * EPS ),
// where EPS is the machine epsilon.
func zget02(trans mat.MatTrans, m, n, nrhs int, a, x, b *mat.CMatrix, rwork *mat.Vector) (resid float64) {
	var cone complex128
	var anorm, bnorm, eps, one, xnorm, zero float64
	var j, n1, n2 int
	var err error

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)

	//     Quick exit if M = 0 or N = 0 or NRHS = 0
	if m <= 0 || n <= 0 || nrhs == 0 {
		resid = zero
		return
	}

	if trans.IsTrans() {
		n1 = n
		n2 = m
	} else {
		n1 = m
		n2 = n
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Zlange('1', m, n, a, rwork)
	if anorm <= zero {
		resid = one / eps
		return
	}

	//     Compute  B - A*X  (or  B - A'*X ) and store in B.
	if err = b.Gemm(trans, NoTrans, n1, nrhs, n2, -cone, a, x, cone); err != nil {
		panic(err)
	}

	//     Compute the maximum over the number of right hand sides of
	//        norm(B - A*X) / ( norm(A) * norm(X) * EPS ) .
	resid = zero
	for j = 1; j <= nrhs; j++ {
		bnorm = b.Off(0, j-1).CVector().Asum(n1, 1)
		xnorm = x.Off(0, j-1).CVector().Asum(n2, 1)
		if xnorm <= zero {
			resid = one / eps
		} else {
			resid = math.Max(resid, ((bnorm/anorm)/xnorm)/eps)
		}
	}

	return
}
