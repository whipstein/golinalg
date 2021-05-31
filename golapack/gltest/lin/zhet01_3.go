package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zhet013 reconstructs a Hermitian indefinite matrix A from its
// block L*D*L' or U*D*U' factorization computed by ZHETRF_RK
// (or ZHETRF_BK) and computes the residual
//    norm( C - A ) / ( N * norm(A) * EPS ),
// where C is the reconstructed matrix and EPS is the machine epsilon.
func Zhet013(uplo byte, n *int, a *mat.CMatrix, lda *int, afac *mat.CMatrix, ldafac *int, e *mat.CVector, ipiv *[]int, c *mat.CMatrix, ldc *int, rwork *mat.Vector, resid *float64) {
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

	//     a) Revert to multiplyers of L
	golapack.Zsyconvfrook(uplo, 'R', n, afac, ldafac, e, ipiv, &info)

	//     1) Determine EPS and the norm of A.
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

	//     2) Initialize C to the identity matrix.
	golapack.Zlaset('F', n, n, &czero, &cone, c, ldc)

	//     3) Call ZLAVHE_ROOK to form the product D * U' (or D * L' ).
	Zlavherook(uplo, 'C', 'N', n, n, afac, ldafac, ipiv, c, ldc, &info)

	//     4) Call ZLAVHE_RK again to multiply by U (or L ).
	Zlavherook(uplo, 'N', 'U', n, n, afac, ldafac, ipiv, c, ldc, &info)

	//     5) Compute the difference  C - A .
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

	//     6) Compute norm( C - A ) / ( N * norm(A) * EPS )
	(*resid) = golapack.Zlanhe('1', uplo, n, c, ldc, rwork)

	if anorm <= zero {
		if (*resid) != zero {
			(*resid) = one / eps
		}
	} else {
		(*resid) = (((*resid) / float64(*n)) / anorm) / eps
	}

	//     b) Convert to factor of L (or U)
	golapack.Zsyconvfrook(uplo, 'C', n, afac, ldafac, e, ipiv, &info)
}
