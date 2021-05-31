package eig

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zbdt02 tests the change of basis C = U' * B by computing the residual
//
//    RESID = norm( B - U * C ) / ( maxint(m,n) * norm(B) * EPS ),
//
// where B and C are M by N matrices, U is an M by M orthogonal matrix,
// and EPS is the machine precision.
func Zbdt02(m, n *int, b *mat.CMatrix, ldb *int, c *mat.CMatrix, ldc *int, u *mat.CMatrix, ldu *int, work *mat.CVector, rwork *mat.Vector, resid *float64) {
	var bnorm, eps, one, realmn, zero float64
	var j int

	zero = 0.0
	one = 1.0

	//     Quick return if possible
	(*resid) = zero
	if (*m) <= 0 || (*n) <= 0 {
		return
	}
	realmn = float64(maxint(*m, *n))
	eps = golapack.Dlamch(Precision)

	//     Compute norm( B - U * C )
	for j = 1; j <= (*n); j++ {
		goblas.Zcopy(m, b.CVector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
		goblas.Zgemv(NoTrans, m, m, toPtrc128(-complex(one, 0)), u, ldu, c.CVector(0, j-1), func() *int { y := 1; return &y }(), toPtrc128(complex(one, 0)), work, func() *int { y := 1; return &y }())
		(*resid) = maxf64(*resid, goblas.Dzasum(m, work, func() *int { y := 1; return &y }()))
	}

	//     Compute norm of B.
	bnorm = golapack.Zlange('1', m, n, b, ldb, rwork)

	if bnorm <= zero {
		if (*resid) != zero {
			(*resid) = one / eps
		}
	} else {
		if bnorm >= (*resid) {
			(*resid) = ((*resid) / bnorm) / (realmn * eps)
		} else {
			if bnorm < one {
				(*resid) = (minf64(*resid, realmn*bnorm) / bnorm) / (realmn * eps)
			} else {
				(*resid) = minf64((*resid)/bnorm, realmn) / (realmn * eps)
			}
		}
	}
}
