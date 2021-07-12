package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dget04 computes the difference between a computed solution and the
// true solution to a system of linear equations.
//
// RESID =  ( norm(X-XACT) * RCOND ) / ( norm(XACT) * EPS ),
// where RCOND is the reciprocal of the condition number and EPS is the
// machine epsilon.
func Dget04(n *int, nrhs *int, x *mat.Matrix, ldx *int, xact *mat.Matrix, ldxact *int, rcond *float64, resid *float64) {
	var diffnm, eps, xnorm, zero float64
	var i, ix, j int

	zero = 0.0

	//     Quick exit if N = 0 or NRHS = 0.
	if (*n) <= 0 || (*nrhs) <= 0 {
		(*resid) = zero
		return
	}

	//     Exit with RESID = 1/EPS if RCOND is invalid.
	eps = golapack.Dlamch(Epsilon)
	if (*rcond) < zero {
		(*resid) = 1.0 / eps
		return
	}

	//     Compute the maximum of
	//        norm(X - XACT) / ( norm(XACT) * EPS )
	//     over all the vectors X and XACT .
	(*resid) = zero
	for j = 1; j <= (*nrhs); j++ {
		ix = goblas.Idamax(*n, xact.Vector(0, j-1, 1))
		xnorm = math.Abs(xact.Get(ix-1, j-1))
		diffnm = zero
		for i = 1; i <= (*n); i++ {
			diffnm = math.Max(diffnm, math.Abs(x.Get(i-1, j-1)-xact.Get(i-1, j-1)))
		}
		if xnorm <= zero {
			if diffnm > zero {
				(*resid) = 1.0 / eps
			}
		} else {
			(*resid) = math.Max(*resid, (diffnm/xnorm)*(*rcond))
		}
	}
	if (*resid)*eps < 1.0 {
		(*resid) = (*resid) / eps
	}
}
