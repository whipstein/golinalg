package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dppt01 reconstructs a symmetric positive definite packed matrix A
// from its L*L' or U'*U factorization and computes the residual
//    norm( L*L' - A ) / ( N * norm(A) * EPS ) or
//    norm( U'*U - A ) / ( N * norm(A) * EPS ),
// where EPS is the machine epsilon.
func dppt01(uplo mat.MatUplo, n int, a, afac, rwork *mat.Vector) (resid float64) {
	var anorm, eps, one, t, zero float64
	var i, k, kc, npp int
	var err error

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0
	if n <= 0 {
		resid = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Dlansp('1', uplo, n, a, rwork)
	if anorm <= zero {
		resid = one / eps
		return
	}

	//     Compute the product U'*U, overwriting U.
	if uplo == Upper {
		kc = (n*(n-1))/2 + 1
		for k = n; k >= 1; k-- {
			//           Compute the (K,K) element of the result.
			t = afac.Off(kc-1).Dot(k, afac.Off(kc-1), 1, 1)
			afac.Set(kc+k-1-1, t)

			//           Compute the rest of column K.
			if k > 1 {
				if err = afac.Off(kc-1).Tpmv(Upper, Trans, NonUnit, k-1, afac, 1); err != nil {
					panic(err)
				}
				kc = kc - (k - 1)
			}
		}

		//     Compute the product L*L', overwriting L.
	} else {
		kc = (n * (n + 1)) / 2
		for k = n; k >= 1; k-- {
			//           Add a multiple of column K of the factor L to each of
			//           columns K+1 through N.
			if k < n {
				if err = afac.Off(kc+n-k).Spr(Lower, n-k, one, afac.Off(kc), 1); err != nil {
					panic(err)
				}
			}

			//           Scale column K by the diagonal element.
			t = afac.Get(kc - 1)
			afac.Off(kc-1).Scal(n-k+1, t, 1)

			kc = kc - (n - k + 2)
		}
	}

	//     Compute the difference  L*L' - A (or U'*U - A).
	npp = n * (n + 1) / 2
	for i = 1; i <= npp; i++ {
		afac.Set(i-1, afac.Get(i-1)-a.Get(i-1))
	}

	//     Compute norm( L*U - A ) / ( N * norm(A) * EPS )
	resid = golapack.Dlansp('1', uplo, n, afac, rwork)

	resid = ((resid / float64(n)) / anorm) / eps

	return
}
