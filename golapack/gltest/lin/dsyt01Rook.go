package lin

import (
	"golinalg/golapack"
	"golinalg/mat"
)

// Dsyt01Rook reconstructs a symmetric indefinite matrix A from its
// block L*D*L' or U*D*U' factorization and computes the residual
//    norm( C - A ) / ( N * norm(A) * EPS ),
// where C is the reconstructed matrix and EPS is the machine epsilon.
func Dsyt01Rook(uplo byte, n *int, a *mat.Matrix, lda *int, afac *mat.Matrix, ldafac *int, ipiv *[]int, c *mat.Matrix, ldc *int, rwork *mat.Vector, resid *float64) {
	var anorm, eps, one, zero float64
	var i, info, j int

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0.
	if (*n) <= 0 {
		(*resid) = zero
		return
	}

	//     Determine EPS and the norm of A.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Dlansy('1', uplo, n, a, lda, rwork)

	//     Initialize C to the identity matrix.
	golapack.Dlaset('F', n, n, &zero, &one, c, ldc)

	//     Call DLAVSY_ROOK to form the product D * U' (or D * L' ).
	DlavsyRook(uplo, 'T', 'N', n, n, afac, ldafac, ipiv, c, ldc, &info)

	//     Call DLAVSY_ROOK again to multiply by U (or L ).
	DlavsyRook(uplo, 'N', 'U', n, n, afac, ldafac, ipiv, c, ldc, &info)

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
	(*resid) = golapack.Dlansy('1', uplo, n, c, ldc, rwork)

	if anorm <= zero {
		if (*resid) != zero {
			(*resid) = one / eps
		}
	} else {
		(*resid) = (((*resid) / float64(*n)) / anorm) / eps
	}
}
