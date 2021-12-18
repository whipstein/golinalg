package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zbdt05 reconstructs a bidiagonal matrix B from its (partial) SVD:
//    S = U' * B * V
// where U and V are orthogonal matrices and S is diagonal.
//
// The test ratio to test the singular value decomposition is
//    RESID = norm( S - U' * B * V ) / ( n * norm(B) * EPS )
// where VT = V' and EPS is the machine precision.
func zbdt05(m, n int, a *mat.CMatrix, s *mat.Vector, ns int, u, vt *mat.CMatrix, work *mat.CVector) (resid float64) {
	var cone, czero complex128
	var anorm, eps, one, zero float64
	var i, j int
	var err error

	dum := vf(1)

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Quick return if possible.
	resid = zero
	if min(m, n) <= 0 || ns <= 0 {
		return
	}

	eps = golapack.Dlamch(Precision)
	anorm = golapack.Zlange('M', m, n, a, dum)

	//     Compute U' * A * V.
	if err = work.Off(1+ns*ns-1).CMatrix(m, opts).Gemm(NoTrans, ConjTrans, m, ns, n, cone, a, vt, czero); err != nil {
		panic(err)
	}
	if err = work.CMatrix(ns, opts).Gemm(ConjTrans, NoTrans, ns, ns, m, -cone, u, work.Off(1+ns*ns-1).CMatrix(m, opts), czero); err != nil {
		panic(err)
	}

	//     norm(S - U' * B * V)
	j = 0
	for i = 1; i <= ns; i++ {
		work.Set(j+i-1, work.Get(j+i-1)+complex(s.Get(i-1), zero))
		resid = math.Max(resid, work.Off(j).Asum(ns, 1))
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
