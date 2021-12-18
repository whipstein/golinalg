package lin

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dqrt16 computes the residual for a solution of a system of linear
// equations  A*x = b  or  A'*x = b:
//    RESID = norm(B - A*X) / ( max(m,n) * norm(A) * norm(X) * EPS ),
// where EPS is the machine epsilon.
func dqrt16(trans mat.MatTrans, m, n, nrhs int, a, x, b *mat.Matrix, rwork *mat.Vector) (resid float64) {
	var anorm, bnorm, eps, one, xnorm, zero float64
	var j, n1, n2 int
	var err error

	zero = 0.0
	one = 1.0

	//     Quick exit if M = 0 or N = 0 or NRHS = 0
	if m <= 0 || n <= 0 || nrhs == 0 {
		resid = zero
		return
	}

	if trans.IsTrans() {
		anorm = golapack.Dlange('I', m, n, a, rwork)
		n1 = n
		n2 = m
	} else {
		anorm = golapack.Dlange('1', m, n, a, rwork)
		n1 = m
		n2 = n
	}

	eps = golapack.Dlamch(Epsilon)

	//     Compute  B - A*X  (or  B - A'*X ) and store in B.
	if err = b.Gemm(trans, NoTrans, n1, nrhs, n2, -one, a, x, one); err != nil {
		panic(err)
	}

	//     Compute the maximum over the number of right hand sides of
	//        norm(B - A*X) / ( max(m,n) * norm(A) * norm(X) * EPS ) .
	resid = zero
	for j = 1; j <= nrhs; j++ {
		bnorm = b.Off(0, j-1).Vector().Asum(n1, 1)
		xnorm = x.Off(0, j-1).Vector().Asum(n2, 1)
		if anorm == zero && bnorm == zero {
			resid = zero
		} else if anorm <= zero || xnorm <= zero {
			resid = one / eps
		} else {
			resid = math.Max(resid, ((bnorm/anorm)/xnorm)/(float64(max(m, n))*eps))
		}
	}

	return
}
