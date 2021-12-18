package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dpot01 reconstructs a symmetric positive definite matrix  A  from
// its L*L' or U'*U factorization and computes the residual
//    norm( L*L' - A ) / ( N * norm(A) * EPS ) or
//    norm( U'*U - A ) / ( N * norm(A) * EPS ),
// where EPS is the machine epsilon.
func dpot01(uplo mat.MatUplo, n int, a, afac *mat.Matrix, rwork *mat.Vector) (resid float64) {
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
		for k = n; k >= 1; k-- {
			//           Compute the (K,K) element of the result.
			t = afac.Off(0, k-1).Vector().Dot(k, afac.Off(0, k-1).Vector(), 1, 1)
			afac.Set(k-1, k-1, t)

			//           Compute the rest of column K.
			if err = afac.Off(0, k-1).Vector().Trmv(Upper, Trans, NonUnit, k-1, afac, 1); err != nil {
				panic(err)
			}

		}

		//     Compute the product L*L', overwriting L.
	} else {
		for k = n; k >= 1; k-- {
			//           Add a multiple of column K of the factor L to each of
			//           columns K+1 through N.
			if k+1 <= n {
				if err = afac.Off(k, k).Syr(Lower, n-k, one, afac.Off(k, k-1).Vector(), 1); err != nil {
					panic(err)
				}
			}

			//           Scale column K by the diagonal element.
			t = afac.Get(k-1, k-1)
			afac.Off(k-1, k-1).Vector().Scal(n-k+1, t, 1)

		}
	}

	//     Compute the difference  L*L' - A (or U'*U - A).
	if uplo == Upper {
		for j = 1; j <= n; j++ {
			for i = 1; i <= j; i++ {
				afac.Set(i-1, j-1, afac.Get(i-1, j-1)-a.Get(i-1, j-1))
			}
		}
	} else {
		for j = 1; j <= n; j++ {
			for i = j; i <= n; i++ {
				afac.Set(i-1, j-1, afac.Get(i-1, j-1)-a.Get(i-1, j-1))
			}
		}
	}

	//     Compute norm( L*U - A ) / ( N * norm(A) * EPS )
	resid = golapack.Dlansy('1', uplo, n, afac, rwork)

	resid = ((resid / float64(n)) / anorm) / eps

	return
}
