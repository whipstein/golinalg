package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zbdt02 tests the change of basis C = U' * B by computing the residual
//
//    RESID = norm( B - U * C ) / ( max(m,n) * norm(B) * EPS ),
//
// where B and C are M by N matrices, U is an M by M orthogonal matrix,
// and EPS is the machine precision.
func Zbdt02(m, n *int, b *mat.CMatrix, ldb *int, c *mat.CMatrix, ldc *int, u *mat.CMatrix, ldu *int, work *mat.CVector, rwork *mat.Vector, resid *float64) {
	var bnorm, eps, one, realmn, zero float64
	var j int
	var err error
	_ = err

	zero = 0.0
	one = 1.0

	//     Quick return if possible
	(*resid) = zero
	if (*m) <= 0 || (*n) <= 0 {
		return
	}
	realmn = float64(max(*m, *n))
	eps = golapack.Dlamch(Precision)

	//     Compute norm( B - U * C )
	for j = 1; j <= (*n); j++ {
		goblas.Zcopy(*m, b.CVector(0, j-1, 1), work.Off(0, 1))
		err = goblas.Zgemv(NoTrans, *m, *m, -complex(one, 0), u, c.CVector(0, j-1, 1), complex(one, 0), work.Off(0, 1))
		(*resid) = math.Max(*resid, goblas.Dzasum(*m, work.Off(0, 1)))
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
				(*resid) = (math.Min(*resid, realmn*bnorm) / bnorm) / (realmn * eps)
			} else {
				(*resid) = math.Min((*resid)/bnorm, realmn) / (realmn * eps)
			}
		}
	}
}
