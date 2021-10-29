package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dget01 reconstructs a matrix A from its L*U factorization and
// computes the residual
//    norm(L*U - A) / ( N * norm(A) * EPS ),
// where EPS is the machine epsilon.
// \endverbatim
func dget01(m, n int, a *mat.Matrix, afac *mat.Matrix, ipiv []int, rwork *mat.Vector) (resid float64) {
	var anorm, eps, one, t, zero float64
	var i, j, k int

	zero = 0.0
	one = 1.0

	//     Quick exit if M = 0 or N = 0.
	if m <= 0 || n <= 0 {
		resid = zero
		return
	}

	//     Determine EPS and the norm of A.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Dlange('1', m, n, a, rwork)

	//     Compute the product L*U and overwrite AFAC with the result.
	//     A column at a time of the product is obtained, starting with
	//     column N.
	for k = n; k >= 1; k-- {
		if k > m {
			if err := goblas.Dtrmv(mat.Lower, mat.NoTrans, mat.Unit, m, afac, afac.Vector(0, k-1, 1)); err != nil {
				panic(err)
			}
		} else {
			//           Compute elements (K+1:M,K)
			t = afac.Get(k-1, k-1)
			if k+1 <= m {
				goblas.Dscal(m-k, t, afac.Vector(k, k-1, 1))
				if err := goblas.Dgemv(mat.NoTrans, m-k, k-1, one, afac.Off(k, 0), afac.Vector(0, k-1, 1), one, afac.Vector(k, k-1, 1)); err != nil {
					panic(err)
				}
			}

			//           Compute the (K,K) element
			afac.Set(k-1, k-1, t+goblas.Ddot(k-1, afac.Vector(k-1, 0), afac.Vector(0, k-1, 1)))

			//           Compute elements (1:K-1,K)
			if err := goblas.Dtrmv(mat.Lower, mat.NoTrans, mat.Unit, k-1, afac, afac.Vector(0, k-1, 1)); err != nil {
				panic(err)
			}
		}
	}
	golapack.Dlaswp(n, afac, 1, min(m, n), ipiv, -1)

	//     Compute the difference  L*U - A  and store in AFAC.
	for j = 1; j <= n; j++ {
		for i = 1; i <= m; i++ {
			afac.Set(i-1, j-1, afac.Get(i-1, j-1)-a.Get(i-1, j-1))
		}
	}

	//     Compute norm( L*U - A ) / ( N * norm(A) * EPS )
	resid = golapack.Dlange('1', m, n, afac, rwork)

	if anorm <= zero {
		if resid != zero {
			resid = one / eps
		}
	} else {
		resid = ((resid / float64(n)) / anorm) / eps
	}

	return
}
