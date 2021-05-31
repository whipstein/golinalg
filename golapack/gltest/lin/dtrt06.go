package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dtrt06 computes a test ratio comparing RCOND (the reciprocal
// condition number of a triangular matrix A) and RCONDC, the estimate
// computed by DTRCON.  Information about the triangular matrix A is
// used if one estimate is zero and the other is non-zero to decide if
// underflow in the estimate is justified.
func Dtrt06(rcond, rcondc *float64, uplo, diag byte, n *int, a *mat.Matrix, lda *int, work *mat.Vector, rat *float64) {
	var anorm, bignum, eps, one, rmax, rmin, smlnum, zero float64

	zero = 0.0
	one = 1.0

	eps = golapack.Dlamch(Epsilon)
	rmax = maxf64(*rcond, *rcondc)
	rmin = minf64(*rcond, *rcondc)

	//     Do the easy cases first.
	if rmin < zero {
		//        Invalid value for RCOND or RCONDC, return 1/EPS.
		(*rat) = one / eps

	} else if rmin > zero {
		//        Both estimates are positive, return RMAX/RMIN - 1.
		(*rat) = rmax/rmin - one

	} else if rmax == zero {
		//        Both estimates zero.
		(*rat) = zero

	} else {
		//        One estimate is zero, the other is non-zero.  If the matrix is
		//        ill-conditioned, return the nonzero estimate multiplied by
		//        1/EPS; if the matrix is badly scaled, return the nonzero
		//        estimate multiplied by BIGNUM/TMAX, where TMAX is the maximum
		//        element in absolute value in A.
		smlnum = golapack.Dlamch(SafeMinimum)
		bignum = one / smlnum
		golapack.Dlabad(&smlnum, &bignum)
		anorm = golapack.Dlantr('M', uplo, diag, n, n, a, lda, work)

		(*rat) = rmax * minf64(bignum/maxf64(one, anorm), one/eps)
	}
}
