package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zhet01 reconstructs a Hermitian indefinite matrix A from its
// block L*D*L' or U*D*U' factorization and computes the residual
//    norm( C - A ) / ( N * norm(A) * EPS ),
// where C is the reconstructed matrix, EPS is the machine epsilon,
// L' is the conjugate transpose of L, and U' is the conjugate transpose
// of U.
func Zhet01(uplo byte, n *int, a *mat.CMatrix, lda *int, afac *mat.CMatrix, ldafac *int, ipiv *[]int, c *mat.CMatrix, ldc *int, rwork *mat.Vector, resid *float64) {
	var cone, czero complex128
	var anorm, eps, one, zero float64
	var i, info, j int

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Quick exit if N = 0.
	if (*n) <= 0 {
		(*resid) = zero
		return
	}

	//     Determine EPS and the norm of A.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Zlanhe('1', uplo, n, a, lda, rwork)

	//     Check the imaginary parts of the diagonal elements and return with
	//     an error code if any are nonzero.
	for j = 1; j <= (*n); j++ {
		if afac.GetIm(j-1, j-1) != zero {
			(*resid) = one / eps
			return
		}
	}

	//     Initialize C to the identity matrix.
	golapack.Zlaset('F', n, n, &czero, &cone, c, ldc)

	//     Call ZLAVHE to form the product D * U' (or D * L' ).
	Zlavhe(uplo, 'C', 'N', n, n, afac, ldafac, ipiv, c, ldc, &info)

	//     Call ZLAVHE again to multiply by U (or L ).
	Zlavhe(uplo, 'N', 'U', n, n, afac, ldafac, ipiv, c, ldc, &info)

	//     Compute the difference  C - A .
	if uplo == 'U' {
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= j-1; i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-a.Get(i-1, j-1))
			}
			c.Set(j-1, j-1, c.Get(j-1, j-1)-a.GetReCmplx(j-1, j-1))
		}
	} else {
		for j = 1; j <= (*n); j++ {
			c.Set(j-1, j-1, c.Get(j-1, j-1)-a.GetReCmplx(j-1, j-1))
			for i = j + 1; i <= (*n); i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-a.Get(i-1, j-1))
			}
		}
	}

	//     Compute norm( C - A ) / ( N * norm(A) * EPS )
	(*resid) = golapack.Zlanhe('1', uplo, n, c, ldc, rwork)

	if anorm <= zero {
		if (*resid) != zero {
			(*resid) = one / eps
		}
	} else {
		(*resid) = (((*resid) / float64(*n)) / anorm) / eps
	}
}
