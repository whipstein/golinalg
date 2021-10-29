package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dpst01 reconstructs a symmetric positive semidefinite matrix A
// from its L or U factors and the permutation matrix P and computes
// the residual
//    norm( P*L*L'*P' - A ) / ( N * norm(A) * EPS ) or
//    norm( P*U'*U*P' - A ) / ( N * norm(A) * EPS ),
// where EPS is the machine epsilon.
func dpst01(uplo mat.MatUplo, n int, a, afac, perm *mat.Matrix, piv []int, rwork *mat.Vector, rank int) (resid float64) {
	var anorm, eps, one, t, zero float64
	var i, j, k int
	var err error

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0.
	if n <= 0 {
		resid = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Dlansy('1', uplo, n, a, rwork)
	if anorm <= zero {
		resid = one / eps
		return
	}

	//     Compute the product U'*U, overwriting U.
	if uplo == Upper {

		if rank < n {
			for j = rank + 1; j <= n; j++ {
				for i = rank + 1; i <= j; i++ {
					afac.Set(i-1, j-1, zero)
				}
			}
		}

		for k = n; k >= 1; k-- {
			//           Compute the (K,K) element of the result.
			t = goblas.Ddot(k, afac.Vector(0, k-1, 1), afac.Vector(0, k-1, 1))
			afac.SetIdx(k-1+(k-1)*afac.Rows, t)

			//           Compute the rest of column K.
			if err = goblas.Dtrmv(mat.Upper, mat.Trans, mat.NonUnit, k-1, afac, afac.Vector(0, k-1, 1)); err != nil {
				panic(err)
			}

		}

		//     Compute the product L*L', overwriting L.
	} else {
		if rank < n {
			for j = rank + 1; j <= n; j++ {
				for i = j; i <= n; i++ {
					afac.Set(i-1, j-1, zero)
				}
			}
		}

		for k = n; k >= 1; k-- {
			//           Add a multiple of column K of the factor L to each of
			//           columns K+1 through N.
			if k+1 <= n {
				if err = goblas.Dsyr(mat.Lower, n-k, one, afac.Vector(k, k-1, 1), afac.Off(k, k)); err != nil {
					panic(err)
				}
			}

			//           Scale column K by the diagonal element.
			t = afac.Get(k-1, k-1)
			goblas.Dscal(n-k+1, t, afac.Vector(k-1, k-1, 1))
		}

	}

	//        Form P*L*L'*P' or P*U'*U*P'
	if uplo == Upper {

		for j = 1; j <= n; j++ {
			for i = 1; i <= n; i++ {
				if piv[i-1] <= piv[j-1] {
					if i <= j {
						perm.Set(piv[i-1]-1, piv[j-1]-1, afac.Get(i-1, j-1))
					} else {
						perm.Set(piv[i-1]-1, piv[j-1]-1, afac.Get(j-1, i-1))
					}
				}
			}
		}

	} else {

		for j = 1; j <= n; j++ {
			for i = 1; i <= n; i++ {
				if piv[i-1] >= piv[j-1] {
					if i >= j {
						perm.Set(piv[i-1]-1, piv[j-1]-1, afac.Get(i-1, j-1))
					} else {
						perm.Set(piv[i-1]-1, piv[j-1]-1, afac.Get(j-1, i-1))
					}
				}
			}
		}

	}

	//     Compute the difference  P*L*L'*P' - A (or P*U'*U*P' - A).
	if uplo == Upper {
		for j = 1; j <= n; j++ {
			for i = 1; i <= j; i++ {
				perm.Set(i-1, j-1, perm.Get(i-1, j-1)-a.GetIdx(i-1+(j-1)*a.Rows))
			}
		}
	} else {
		for j = 1; j <= n; j++ {
			for i = j; i <= n; i++ {
				perm.Set(i-1, j-1, perm.Get(i-1, j-1)-a.GetIdx(i-1+(j-1)*a.Rows))
			}
		}
	}

	//     Compute norm( P*L*L'P - A ) / ( N * norm(A) * EPS ), or
	//     ( P*U'*U*P' - A )/ ( N * norm(A) * EPS ).
	resid = golapack.Dlansy('1', uplo, n, perm, rwork)

	resid = ((resid / float64(n)) / anorm) / eps

	return
}
