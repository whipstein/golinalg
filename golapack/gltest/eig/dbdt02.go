package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dbdt02 tests the change of basis C = U' * B by computing the residual
//
//    RESID = norm( B - U * C ) / ( max(m,n) * norm(B) * EPS ),
//
// where B and C are M by N matrices, U is an M by M orthogonal matrix,
// and EPS is the machine precision.
func dbdt02(m, n int, b, c, u *mat.Matrix, work *mat.Vector) (resid float64) {
	var bnorm, eps, one, realmn, zero float64
	var j int
	var err error

	zero = 0.0
	one = 1.0

	//     Quick return if possible
	resid = zero
	if m <= 0 || n <= 0 {
		return
	}
	realmn = float64(max(m, n))
	eps = golapack.Dlamch(Precision)

	//     Compute norm( B - U * C )
	for j = 1; j <= n; j++ {
		goblas.Dcopy(m, b.Vector(0, j-1, 1), work.Off(0, 1))
		if err = goblas.Dgemv(NoTrans, m, m, -one, u, c.Vector(0, j-1, 1), one, work.Off(0, 1)); err != nil {
			panic(err)
		}
		resid = math.Max(resid, goblas.Dasum(m, work.Off(0, 1)))
	}

	//     Compute norm of B.
	bnorm = golapack.Dlange('1', m, n, b, work)

	if bnorm <= zero {
		if resid != zero {
			resid = one / eps
		}
	} else {
		if bnorm >= resid {
			resid = (resid / bnorm) / (realmn * eps)
		} else {
			if bnorm < one {
				resid = (math.Min(resid, realmn*bnorm) / bnorm) / (realmn * eps)
			} else {
				resid = math.Min(resid/bnorm, realmn) / (realmn * eps)
			}
		}
	}
	return
}
