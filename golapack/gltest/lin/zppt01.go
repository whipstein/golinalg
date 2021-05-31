package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zppt01 reconstructs a Hermitian positive definite packed matrix A
// from its L*L' or U'*U factorization and computes the residual
//    norm( L*L' - A ) / ( N * norm(A) * EPS ) or
//    norm( U'*U - A ) / ( N * norm(A) * EPS ),
// where EPS is the machine epsilon, L' is the conjugate transpose of
// L, and U' is the conjugate transpose of U.
func Zppt01(uplo byte, n *int, a, afac *mat.CVector, rwork *mat.Vector, resid *float64) {
	var tc complex128
	var anorm, eps, one, tr, zero float64
	var i, k, kc int

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0
	if (*n) <= 0 {
		(*resid) = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Zlanhp('1', uplo, n, a, rwork)
	if anorm <= zero {
		(*resid) = one / eps
		return
	}

	//     Check the imaginary parts of the diagonal elements and return with
	//     an error code if any are nonzero.
	kc = 1
	if uplo == 'U' {
		for k = 1; k <= (*n); k++ {
			if imag(afac.Get(kc-1)) != zero {
				(*resid) = one / eps
				return
			}
			kc = kc + k + 1
		}
	} else {
		for k = 1; k <= (*n); k++ {
			if imag(afac.Get(kc-1)) != zero {
				(*resid) = one / eps
				return
			}
			kc = kc + (*n) - k + 1
		}
	}

	//     Compute the product U'*U, overwriting U.
	if uplo == 'U' {
		kc = ((*n)*((*n)-1))/2 + 1
		for k = (*n); k >= 1; k-- {
			//           Compute the (K,K) element of the result.
			tr = real(goblas.Zdotc(&k, afac.Off(kc-1), func() *int { y := 1; return &y }(), afac.Off(kc-1), func() *int { y := 1; return &y }()))
			afac.SetRe(kc+k-1-1, tr)

			//           Compute the rest of column K.
			if k > 1 {
				goblas.Ztpmv(Upper, ConjTrans, NonUnit, toPtr(k-1), afac, afac.Off(kc-1), func() *int { y := 1; return &y }())
				kc = kc - (k - 1)
			}
		}

		//        Compute the difference  L*L' - A
		kc = 1
		for k = 1; k <= (*n); k++ {
			for i = 1; i <= k-1; i++ {
				afac.Set(kc+i-1-1, afac.Get(kc+i-1-1)-a.Get(kc+i-1-1))
			}
			afac.Set(kc+k-1-1, afac.Get(kc+k-1-1)-a.GetReCmplx(kc+k-1-1))
			kc = kc + k
		}

		//     Compute the product L*L', overwriting L.
	} else {
		kc = ((*n) * ((*n) + 1)) / 2
		for k = (*n); k >= 1; k-- {

			//           Add a multiple of column K of the factor L to each of
			//           columns K+1 through N.
			if k < (*n) {
				goblas.Zhpr(Lower, toPtr((*n)-k), &one, afac.Off(kc+1-1), func() *int { y := 1; return &y }(), afac.Off(kc+(*n)-k+1-1))
			}

			//           Scale column K by the diagonal element.
			tc = afac.Get(kc - 1)
			goblas.Zscal(toPtr((*n)-k+1), &tc, afac.Off(kc-1), func() *int { y := 1; return &y }())

			kc = kc - ((*n) - k + 2)
		}

		//        Compute the difference  U'*U - A
		kc = 1
		for k = 1; k <= (*n); k++ {
			afac.Set(kc-1, afac.Get(kc-1)-a.GetReCmplx(kc-1))
			for i = k + 1; i <= (*n); i++ {
				afac.Set(kc+i-k-1, afac.Get(kc+i-k-1)-a.Get(kc+i-k-1))
			}
			kc = kc + (*n) - k + 1
		}
	}

	//     Compute norm( L*U - A ) / ( N * norm(A) * EPS )
	(*resid) = golapack.Zlanhp('1', uplo, n, afac, rwork)

	(*resid) = (((*resid) / float64(*n)) / anorm) / eps
}
