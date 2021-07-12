package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dget02 computes the residual for a solution of a system of linear
// equations  A*x = b  or  A'*x = b:
//    RESID = norm(B - A*X) / ( norm(A) * norm(X) * EPS ),
// where EPS is the machine epsilon.
func Dget02(trans byte, m *int, n *int, nrhs *int, a *mat.Matrix, lda *int, x *mat.Matrix, ldx *int, b *mat.Matrix, ldb *int, rwork *mat.Vector, resid *float64) {
	var anorm, bnorm, eps, one, xnorm, zero float64
	var j, n1, n2 int
	var err error
	_ = err

	zero = 0.0
	one = 1.0

	//     Quick exit if M = 0 or N = 0 or NRHS = 0
	if (*m) <= 0 || (*n) <= 0 || (*nrhs) == 0 {
		(*resid) = zero
		return
	}

	_trans := mat.NoTrans
	if trans == 'T' || trans == 'C' {
		n1 = (*n)
		n2 = (*m)
		_trans = mat.Trans
		if trans == 'C' {
			_trans = mat.ConjTrans
		}
	} else {
		n1 = (*m)
		n2 = (*n)
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Dlange('1', m, n, a, lda, rwork)
	if anorm <= zero {
		(*resid) = one / eps
		return
	}

	//     Compute  B - A*X  (or  B - A'*X ) and store in B.
	err = goblas.Dgemm(_trans, mat.NoTrans, n1, *nrhs, n2, -one, a, x, one, b)

	//     Compute the maximum over the number of right hand sides of
	//        norm(B - A*X) / ( norm(A) * norm(X) * EPS ) .
	(*resid) = zero
	for j = 1; j <= (*nrhs); j++ {
		bnorm = goblas.Dasum(n1, b.Vector(0, j-1, 1))
		xnorm = goblas.Dasum(n2, x.Vector(0, j-1, 1))
		if xnorm <= zero {
			(*resid) = one / eps
		} else {
			(*resid) = math.Max(*resid, ((bnorm/anorm)/xnorm)/eps)
		}
	}
}
