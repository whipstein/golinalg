package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zsyt01rook reconstructs a complex symmetric indefinite matrix A from its
// block L*D*L' or U*D*U' factorization and computes the residual
//    norm( C - A ) / ( N * norm(A) * EPS ),
// where C is the reconstructed matrix, EPS is the machine epsilon,
// L' is the transpose of L, and U' is the transpose of U.
func Zsyt01rook(uplo byte, n *int, a *mat.CMatrix, lda *int, afac *mat.CMatrix, ldafac *int, ipiv *[]int, c *mat.CMatrix, ldc *int, rwork *mat.Vector, resid *float64) {
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
	anorm = golapack.Zlansy('1', uplo, n, a, lda, rwork)

	//     Initialize C to the identity matrix.
	golapack.Zlaset('F', n, n, &czero, &cone, c, ldc)

	//     Call ZLAVSY_ROOK to form the product D * U' (or D * L' ).
	Zlavsyrook(uplo, 'T', 'N', n, n, afac, ldafac, ipiv, c, ldc, &info)

	//     Call ZLAVSY_ROOK again to multiply by U (or L ).
	Zlavsyrook(uplo, 'N', 'U', n, n, afac, ldafac, ipiv, c, ldc, &info)

	//     Compute the difference  C - A .
	if uplo == 'U' {
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= j; i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-a.Get(i-1, j-1))
			}
		}
	} else {
		for j = 1; j <= (*n); j++ {
			for i = j; i <= (*n); i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-a.Get(i-1, j-1))
			}
		}
	}

	//     Compute norm( C - A ) / ( N * norm(A) * EPS )
	(*resid) = golapack.Zlansy('1', uplo, n, c, ldc, rwork)

	if anorm <= zero {
		if (*resid) != zero {
			(*resid) = one / eps
		}
	} else {
		(*resid) = (((*resid) / float64(*n)) / anorm) / eps
	}
}
