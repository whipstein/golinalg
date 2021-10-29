package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zpot01 reconstructs a Hermitian positive definite matrix  A  from
// its L*L' or U'*U factorization and computes the residual
//    norm( L*L' - A ) / ( N * norm(A) * EPS ) or
//    norm( U'*U - A ) / ( N * norm(A) * EPS ),
// where EPS is the machine epsilon, L' is the conjugate transpose of L,
// and U' is the conjugate transpose of U.
func zpot01(uplo mat.MatUplo, n int, a, afac *mat.CMatrix, rwork *mat.Vector) (resid float64) {
	var tc complex128
	var anorm, eps, one, tr, zero float64
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
	anorm = golapack.Zlanhe('1', uplo, n, a, rwork)
	if anorm <= zero {
		resid = one / eps
		return
	}

	//     Check the imaginary parts of the diagonal elements and return with
	//     an error code if any are nonzero.
	for j = 1; j <= n; j++ {
		if imag(afac.Get(j-1, j-1)) != zero {
			resid = one / eps
			return
		}
	}

	//     Compute the product U'*U, overwriting U.
	if uplo == Upper {
		for k = n; k >= 1; k-- {
			//           Compute the (K,K) element of the result.
			tr = real(goblas.Zdotc(k, afac.CVector(0, k-1, 1), afac.CVector(0, k-1, 1)))
			afac.SetRe(k-1, k-1, tr)

			//           Compute the rest of column K.
			if err = goblas.Ztrmv(Upper, ConjTrans, NonUnit, k-1, afac, afac.CVector(0, k-1, 1)); err != nil {
				panic(err)
			}

		}

		//     Compute the product L*L', overwriting L.
	} else {
		for k = n; k >= 1; k-- {
			//           Add a multiple of column K of the factor L to each of
			//           columns K+1 through N.
			if k+1 <= n {
				if err = goblas.Zher(Lower, n-k, one, afac.CVector(k, k-1, 1), afac.Off(k, k)); err != nil {
					panic(err)
				}
			}

			//           Scale column K by the diagonal element.
			tc = afac.Get(k-1, k-1)
			goblas.Zscal(n-k+1, tc, afac.CVector(k-1, k-1, 1))

		}
	}

	//     Compute the difference  L*L' - A (or U'*U - A).
	if uplo == Upper {
		for j = 1; j <= n; j++ {
			for i = 1; i <= j-1; i++ {
				afac.Set(i-1, j-1, afac.Get(i-1, j-1)-a.Get(i-1, j-1))
			}
			afac.Set(j-1, j-1, afac.Get(j-1, j-1)-a.GetReCmplx(j-1, j-1))
		}
	} else {
		for j = 1; j <= n; j++ {
			afac.Set(j-1, j-1, afac.Get(j-1, j-1)-a.GetReCmplx(j-1, j-1))
			for i = j + 1; i <= n; i++ {
				afac.Set(i-1, j-1, afac.Get(i-1, j-1)-a.Get(i-1, j-1))
			}
		}
	}

	//     Compute norm( L*U - A ) / ( N * norm(A) * EPS )
	resid = golapack.Zlanhe('1', uplo, n, afac, rwork)

	resid = ((resid / float64(n)) / anorm) / eps

	return
}
