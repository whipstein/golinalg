package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dgtt02 computes the residual for the solution to a tridiagonal
// system of equations:
//    RESID = norm(B - op(A)*X) / (norm(A) * norm(X) * EPS),
// where EPS is the machine epsilon.
func Dgtt02(trans byte, n, nrhs *int, dl, d, du *mat.Vector, x *mat.Matrix, ldx *int, b *mat.Matrix, ldb *int, resid *float64) {
	var anorm, bnorm, eps, one, xnorm, zero float64
	var j int

	one = 1.0
	zero = 0.0

	//     Quick exit if N = 0 or NRHS = 0
	(*resid) = zero
	if (*n) <= 0 || (*nrhs) == 0 {
		return
	}

	//     Compute the maximum over the number of right hand sides of
	//        norm(B - op(A)*X) / ( norm(A) * norm(X) * EPS ).
	if trans == 'N' {
		anorm = golapack.Dlangt('1', n, dl, d, du)
	} else {
		anorm = golapack.Dlangt('I', n, dl, d, du)
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	if anorm <= zero {
		(*resid) = one / eps
		return
	}

	//     Compute B - op(A)*X.
	golapack.Dlagtm(trans, n, nrhs, func() *float64 { y := -one; return &y }(), dl, d, du, x, ldx, &one, b, ldb)

	for j = 1; j <= (*nrhs); j++ {
		bnorm = goblas.Dasum(*n, b.Vector(0, j-1, 1))
		xnorm = goblas.Dasum(*n, x.Vector(0, j-1, 1))
		if xnorm <= zero {
			(*resid) = one / eps
		} else {
			(*resid) = math.Max(*resid, ((bnorm/anorm)/xnorm)/eps)
		}
	}
}
