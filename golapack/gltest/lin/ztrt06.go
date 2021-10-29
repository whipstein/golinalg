package lin

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// ztrt06 computes a test ratio comparing RCOND (the reciprocal
// condition number of a triangular matrix A) and RCONDC, the estimate
// computed by ZTRCON.  Information about the triangular matrix A is
// used if one estimate is zero and the other is non-zero to decide if
// underflow in the estimate is justified.
func ztrt06(rcond, rcondc float64, uplo mat.MatUplo, diag mat.MatDiag, n int, a *mat.CMatrix, rwork *mat.Vector) (rat float64) {
	var anorm, bignum, eps, one, rmax, rmin, zero float64

	zero = 0.0
	one = 1.0

	eps = golapack.Dlamch(Epsilon)
	rmax = math.Max(rcond, rcondc)
	rmin = math.Min(rcond, rcondc)

	//     Do the easy cases first.
	if rmin < zero {
		//        Invalid value for RCOND or RCONDC, return 1/EPS.
		rat = one / eps

	} else if rmin > zero {
		//        Both estimates are positive, return RMAX/RMIN - 1.
		rat = rmax/rmin - one

	} else if rmax == zero {
		//        Both estimates zero.
		rat = zero

	} else {
		//        One estimate is zero, the other is non-zero.  If the matrix is
		//        ill-conditioned, return the nonzero estimate multiplied by
		//        1/EPS; if the matrix is badly scaled, return the nonzero
		//        estimate multiplied by BIGNUM/TMAX, where TMAX is the maximum
		//        element in absolute value in A.
		bignum = one / golapack.Dlamch(SafeMinimum)
		anorm = golapack.Zlantr('M', uplo, diag, n, n, a, rwork)

		rat = rmax * math.Min(bignum/math.Max(one, anorm), one/eps)
	}

	return
}
