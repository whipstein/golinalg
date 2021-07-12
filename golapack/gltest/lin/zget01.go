package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zget01 reconstructs a matrix A from its L*U factorization and
// computes the residual
//    norm(L*U - A) / ( N * norm(A) * EPS ),
// where EPS is the machine epsilon.
func Zget01(m, n *int, a *mat.CMatrix, lda *int, afac *mat.CMatrix, ldafac *int, ipiv *[]int, rwork *mat.Vector, resid *float64) {
	var cone, t complex128
	var anorm, eps, one, zero float64
	var i, j, k int
	var err error
	_ = err

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)

	//     Quick exit if M = 0 or N = 0.
	if (*m) <= 0 || (*n) <= 0 {
		(*resid) = zero
		return
	}

	//     Determine EPS and the norm of A.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Zlange('1', m, n, a, lda, rwork)

	//     Compute the product L*U and overwrite AFAC with the result.
	//     A column at a time of the product is obtained, starting with
	//     column N.
	for k = (*n); k >= 1; k-- {
		if k > (*m) {
			err = goblas.Ztrmv(Lower, NoTrans, Unit, *m, afac, afac.CVector(0, k-1, 1))
		} else {
			//           Compute elements (K+1:M,K)
			t = afac.Get(k-1, k-1)
			if k+1 <= (*m) {
				goblas.Zscal((*m)-k, t, afac.CVector(k, k-1, 1))
				err = goblas.Zgemv(NoTrans, (*m)-k, k-1, cone, afac.Off(k, 0), afac.CVector(0, k-1, 1), cone, afac.CVector(k, k-1, 1))
			}

			//           Compute the (K,K) element
			afac.Set(k-1, k-1, t+goblas.Zdotu(k-1, afac.CVector(k-1, 0), afac.CVector(0, k-1, 1)))

			//           Compute elements (1:K-1,K)
			err = goblas.Ztrmv(Lower, NoTrans, Unit, k-1, afac, afac.CVector(0, k-1, 1))
		}
	}
	golapack.Zlaswp(n, afac, ldafac, func() *int { y := 1; return &y }(), toPtr(min(*m, *n)), ipiv, toPtr(-1))

	//     Compute the difference  L*U - A  and store in AFAC.
	for j = 1; j <= (*n); j++ {
		for i = 1; i <= (*m); i++ {
			afac.Set(i-1, j-1, afac.Get(i-1, j-1)-a.Get(i-1, j-1))
		}
	}

	//     Compute norm( L*U - A ) / ( N * norm(A) * EPS )
	(*resid) = golapack.Zlange('1', m, n, afac, ldafac, rwork)

	if anorm <= zero {
		if (*resid) != zero {
			(*resid) = one / eps
		}
	} else {
		(*resid) = (((*resid) / float64(*n)) / anorm) / eps
	}
}
