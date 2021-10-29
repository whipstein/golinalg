package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dbdt05 reconstructs a bidiagonal matrix B from its (partial) SVD:
//    S = U' * B * V
// where U and V are orthogonal matrices and S is diagonal.
//
// The test ratio to test the singular value decomposition is
//    RESID = norm( S - U' * B * V ) / ( n * norm(B) * EPS )
// where VT = V' and EPS is the machine precision.
func dbdt05(m, n int, a *mat.Matrix, s *mat.Vector, ns int, u, vt *mat.Matrix, work *mat.Vector) (resid float64) {
	var anorm, eps, one, zero float64
	var i, j int
	var err error

	zero = 0.0
	one = 1.0

	//     Quick return if possible.
	resid = zero
	if min(m, n) <= 0 || ns <= 0 {
		return
	}

	eps = golapack.Dlamch(Precision)
	anorm = golapack.Dlange('M', m, n, a, work)

	//     Compute U' * A * V.
	if err = goblas.Dgemm(NoTrans, Trans, m, ns, n, one, a, vt, zero, work.MatrixOff(1+ns*ns-1, m, opts)); err != nil {
		panic(err)
	}
	if err = goblas.Dgemm(Trans, NoTrans, ns, ns, m, -one, u, work.MatrixOff(1+ns*ns-1, m, opts), zero, work.Matrix(ns, opts)); err != nil {
		panic(err)
	}

	//     norm(S - U' * B * V)
	j = 0
	for i = 1; i <= ns; i++ {
		work.Set(j+i-1, work.Get(j+i-1)+s.Get(i-1))
		resid = math.Max(resid, goblas.Dasum(ns, work.Off(j, 1)))
		j = j + ns
	}

	if anorm <= zero {
		if resid != zero {
			resid = one / eps
		}
	} else {
		if anorm >= resid {
			resid = (resid / anorm) / (float64(n) * eps)
		} else {
			if anorm < one {
				resid = (math.Min(resid, float64(n)*anorm) / anorm) / (float64(n) * eps)
			} else {
				resid = math.Min(resid/anorm, float64(n)) / (float64(n) * eps)
			}
		}
	}

	return
}
