package lin

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zgtt02 computes the residual for the solution to a tridiagonal
// system of equations:
//    RESID = norm(B - op(A)*X) / (norm(A) * norm(X) * EPS),
// where EPS is the machine epsilon.
func zgtt02(trans mat.MatTrans, n, nrhs int, dl, d, du *mat.CVector, x, b *mat.CMatrix) (resid float64) {
	var anorm, bnorm, eps, one, xnorm, zero float64
	var j int

	one = 1.0
	zero = 0.0

	//     Quick exit if N = 0 or NRHS = 0
	resid = zero
	if n <= 0 || nrhs == 0 {
		return
	}

	//     Compute the maximum over the number of right hand sides of
	//        norm(B - op(A)*X) / ( norm(A) * norm(X) * EPS ).
	if trans == NoTrans {
		anorm = golapack.Zlangt('1', n, dl, d, du)
	} else {
		anorm = golapack.Zlangt('I', n, dl, d, du)
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	if anorm <= zero {
		resid = one / eps
		return
	}

	//     Compute B - op(A)*X.
	golapack.Zlagtm(trans, n, nrhs, -one, dl, d, du, x, one, b)

	for j = 1; j <= nrhs; j++ {
		bnorm = b.Off(0, j-1).CVector().Asum(n, 1)
		xnorm = x.Off(0, j-1).CVector().Asum(n, 1)
		if xnorm <= zero {
			resid = one / eps
		} else {
			resid = math.Max(resid, ((bnorm/anorm)/xnorm)/eps)
		}
	}

	return
}
