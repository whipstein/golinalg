package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dppt01 reconstructs a symmetric positive definite packed matrix A
// from its L*L' or U'*U factorization and computes the residual
//    norm( L*L' - A ) / ( N * norm(A) * EPS ) or
//    norm( U'*U - A ) / ( N * norm(A) * EPS ),
// where EPS is the machine epsilon.
func Dppt01(uplo byte, n *int, a, afac, rwork *mat.Vector, resid *float64) {
	var anorm, eps, one, t, zero float64
	var i, k, kc, npp int
	var err error
	_ = err

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0
	if (*n) <= 0 {
		(*resid) = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Dlansp('1', uplo, n, a, rwork)
	if anorm <= zero {
		(*resid) = one / eps
		return
	}

	//     Compute the product U'*U, overwriting U.
	if uplo == 'U' {
		kc = ((*n)*((*n)-1))/2 + 1
		for k = (*n); k >= 1; k-- {
			//           Compute the (K,K) element of the result.
			t = goblas.Ddot(k, afac.Off(kc-1), 1, afac.Off(kc-1), 1)
			afac.Set(kc+k-1-1, t)

			//           Compute the rest of column K.
			if k > 1 {
				err = goblas.Dtpmv(mat.Upper, mat.Trans, mat.NonUnit, k-1, afac, afac.Off(kc-1), 1)
				kc = kc - (k - 1)
			}
		}

		//     Compute the product L*L', overwriting L.
	} else {
		kc = ((*n) * ((*n) + 1)) / 2
		for k = (*n); k >= 1; k-- {
			//           Add a multiple of column K of the factor L to each of
			//           columns K+1 through N.
			if k < (*n) {
				err = goblas.Dspr(mat.Lower, (*n)-k, one, afac.Off(kc+1-1), 1, afac.Off(kc+(*n)-k+1-1))
			}

			//           Scale column K by the diagonal element.
			t = afac.Get(kc - 1)
			goblas.Dscal((*n)-k+1, t, afac.Off(kc-1), 1)

			kc = kc - ((*n) - k + 2)
		}
	}

	//     Compute the difference  L*L' - A (or U'*U - A).
	npp = (*n) * ((*n) + 1) / 2
	for i = 1; i <= npp; i++ {
		afac.Set(i-1, afac.Get(i-1)-a.Get(i-1))
	}

	//     Compute norm( L*U - A ) / ( N * norm(A) * EPS )
	(*resid) = golapack.Dlansp('1', uplo, n, afac, rwork)

	(*resid) = (((*resid) / float64(*n)) / anorm) / eps
}
