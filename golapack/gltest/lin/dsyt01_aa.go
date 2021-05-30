package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
)

// Dsyt01Aa reconstructs a symmetric indefinite matrix A from its
// block L*D*L' or U*D*U' factorization and computes the residual
//    norm( C - A ) / ( N * norm(A) * EPS ),
// where C is the reconstructed matrix and EPS is the machine epsilon.
func Dsyt01Aa(uplo byte, n *int, a *mat.Matrix, lda *int, afac *mat.Matrix, ldafac *int, ipiv *[]int, c *mat.Matrix, ldc *int, rwork *mat.Vector, resid *float64) {
	var anorm, eps, one, zero float64
	var i, j int

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

	//     Initialize C to the tridiagonal matrix T.
	golapack.Dlaset('F', n, n, &zero, &zero, c, ldc)
	golapack.Dlacpy('F', toPtr(1), n, afac.Off(0, 0).UpdateRows((*ldafac)+1), toPtr((*ldafac)+1), c.Off(0, 0).UpdateRows((*ldc)+1), toPtr((*ldc)+1))
	if (*n) > 1 {
		if uplo == 'U' {
			golapack.Dlacpy('F', toPtr(1), toPtr((*n)-1), afac.Off(0, 1).UpdateRows((*ldafac)+1), toPtr((*ldafac)+1), c.Off(0, 1).UpdateRows((*ldc)+1), toPtr((*ldc)+1))
			golapack.Dlacpy('F', toPtr(1), toPtr((*n)-1), afac.Off(0, 1).UpdateRows((*ldafac)+1), toPtr((*ldafac)+1), c.Off(1, 0).UpdateRows((*ldc)+1), toPtr((*ldc)+1))
		} else {
			golapack.Dlacpy('F', toPtr(1), toPtr((*n)-1), afac.Off(1, 0).UpdateRows((*ldafac)+1), toPtr((*ldafac)+1), c.Off(0, 1).UpdateRows((*ldc)+1), toPtr((*ldc)+1))
			golapack.Dlacpy('F', toPtr(1), toPtr((*n)-1), afac.Off(1, 0).UpdateRows((*ldafac)+1), toPtr((*ldafac)+1), c.Off(1, 0).UpdateRows((*ldc)+1), toPtr((*ldc)+1))
		}

		//        Call DTRMM to form the product U' * D (or L * D ).
		if uplo == 'U' {
			goblas.Dtrmm(mat.Left, mat.UploByte(uplo), mat.Trans, mat.Unit, toPtr((*n)-1), n, &one, afac.Off(0, 1), ldafac, c.Off(1, 0), ldc)
		} else {
			goblas.Dtrmm(mat.Left, mat.UploByte(uplo), mat.NoTrans, mat.Unit, toPtr((*n)-1), n, &one, afac.Off(1, 0), ldafac, c.Off(1, 0), ldc)
		}

		//        Call DTRMM again to multiply by U (or L ).
		if uplo == 'U' {
			goblas.Dtrmm(mat.Right, mat.UploByte(uplo), mat.NoTrans, mat.Unit, n, toPtr((*n)-1), &one, afac.Off(0, 1), ldafac, c.Off(0, 1), ldc)
		} else {
			goblas.Dtrmm(mat.Right, mat.UploByte(uplo), mat.Trans, mat.Unit, n, toPtr((*n)-1), &one, afac.Off(1, 0), ldafac, c.Off(0, 1), ldc)
		}
	}

	//     Apply symmetric pivots
	for j = (*n); j >= 1; j-- {
		i = (*ipiv)[j-1]
		if i != j {
			goblas.Dswap(n, c.Vector(j-1, 0), ldc, c.Vector(i-1, 0), ldc)
		}
	}
	for j = (*n); j >= 1; j-- {
		i = (*ipiv)[j-1]
		if i != j {
			goblas.Dswap(n, c.Vector(0, j-1), toPtr(1), c.Vector(0, i-1), toPtr(1))
		}
	}

	//
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
