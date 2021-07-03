package eig

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zbdt05 reconstructs a bidiagonal matrix B from its (partial) SVD:
//    S = U' * B * V
// where U and V are orthogonal matrices and S is diagonal.
//
// The test ratio to test the singular value decomposition is
//    RESID = norm( S - U' * B * V ) / ( n * norm(B) * EPS )
// where VT = V' and EPS is the machine precision.
func Zbdt05(m, n *int, a *mat.CMatrix, lda *int, s *mat.Vector, ns *int, u *mat.CVector, ldu *int, vt *mat.CMatrix, ldvt *int, work *mat.CVector, resid *float64) {
	var cone, czero complex128
	var anorm, eps, one, zero float64
	var i, j int
	var err error
	_ = err

	dum := vf(1)

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Quick return if possible.
	(*resid) = zero
	if minint(*m, *n) <= 0 || (*ns) <= 0 {
		return
	}

	eps = golapack.Dlamch(Precision)
	anorm = golapack.Zlange('M', m, n, a, lda, dum)

	//     Compute U' * A * V.
	err = goblas.Zgemm(NoTrans, ConjTrans, *m, *ns, *n, cone, a, *lda, vt, *ldvt, czero, work.CMatrixOff(1+(*ns)*(*ns)-1, *m, opts), *m)
	err = goblas.Zgemm(ConjTrans, NoTrans, *ns, *ns, *m, -cone, u.CMatrix(*ldu, opts), *ldu, work.CMatrixOff(1+(*ns)*(*ns)-1, *m, opts), *m, czero, work.CMatrix(*ns, opts), *ns)

	//     norm(S - U' * B * V)
	j = 0
	for i = 1; i <= (*ns); i++ {
		work.Set(j+i-1, work.Get(j+i-1)+complex(s.Get(i-1), zero))
		(*resid) = maxf64(*resid, goblas.Dzasum(*ns, work.Off(j+1-1), 1))
		j = j + (*ns)
	}

	if anorm <= zero {
		if (*resid) != zero {
			(*resid) = one / eps
		}
	} else {
		if anorm >= (*resid) {
			(*resid) = ((*resid) / anorm) / (float64(*n) * eps)
		} else {
			if anorm < one {
				(*resid) = (minf64(*resid, float64(*n)*anorm) / anorm) / (float64(*n) * eps)
			} else {
				(*resid) = minf64((*resid)/anorm, float64(*n)) / (float64(*n) * eps)
			}
		}
	}
}
