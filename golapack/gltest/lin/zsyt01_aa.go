package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zsyt01 reconstructs a hermitian indefinite matrix A from its
// block L*D*L' or U*D*U' factorization and computes the residual
//    norm( C - A ) / ( N * norm(A) * EPS ),
// where C is the reconstructed matrix and EPS is the machine epsilon.
func Zsyt01aa(uplo byte, n *int, a *mat.CMatrix, lda *int, afac *mat.CMatrix, ldafac *int, ipiv *[]int, c *mat.CMatrix, ldc *int, rwork *mat.Vector, resid *float64) {
	var cone, czero complex128
	var anorm, eps, one, zero float64
	var i, j int

	zero = 0.0
	one = 1.0
	czero = 0.0
	cone = 1.0

	//     Quick exit if N = 0.
	if (*n) <= 0 {
		(*resid) = zero
		return
	}

	//     Determine EPS and the norm of A.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Zlansy('1', uplo, n, a, lda, rwork)

	//     Initialize C to the tridiagonal matrix T.
	golapack.Zlaset('F', n, n, &czero, &czero, c, ldc)
	golapack.Zlacpy('F', func() *int { y := 1; return &y }(), n, afac.Off(0, 0).UpdateRows((*ldafac)+1), toPtr((*ldafac)+1), c.Off(0, 0).UpdateRows((*ldc)+1), toPtr((*ldc)+1))
	if (*n) > 1 {
		if uplo == 'U' {
			golapack.Zlacpy('F', func() *int { y := 1; return &y }(), toPtr((*n)-1), afac.Off(0, 1).UpdateRows((*ldafac)+1), toPtr((*ldafac)+1), c.Off(0, 1).UpdateRows((*ldc)+1), toPtr((*ldc)+1))
			golapack.Zlacpy('F', func() *int { y := 1; return &y }(), toPtr((*n)-1), afac.Off(0, 1).UpdateRows((*ldafac)+1), toPtr((*ldafac)+1), c.Off(1, 0).UpdateRows((*ldc)+1), toPtr((*ldc)+1))
		} else {
			golapack.Zlacpy('F', func() *int { y := 1; return &y }(), toPtr((*n)-1), afac.Off(1, 0).UpdateRows((*ldafac)+1), toPtr((*ldafac)+1), c.Off(0, 1).UpdateRows((*ldc)+1), toPtr((*ldc)+1))
			golapack.Zlacpy('F', func() *int { y := 1; return &y }(), toPtr((*n)-1), afac.Off(1, 0).UpdateRows((*ldafac)+1), toPtr((*ldafac)+1), c.Off(1, 0).UpdateRows((*ldc)+1), toPtr((*ldc)+1))
		}

		//        Call ZTRMM to form the product U' * D (or L * D ).
		if uplo == 'U' {
			goblas.Ztrmm(Left, mat.UploByte(uplo), Trans, Unit, toPtr((*n)-1), n, &cone, afac.Off(0, 1), ldafac, c.Off(1, 0), ldc)
		} else {
			goblas.Ztrmm(Left, mat.UploByte(uplo), NoTrans, Unit, toPtr((*n)-1), n, &cone, afac.Off(1, 0), ldafac, c.Off(1, 0), ldc)
		}

		//        Call ZTRMM again to multiply by U (or L ).
		if uplo == 'U' {
			goblas.Ztrmm(Right, mat.UploByte(uplo), NoTrans, Unit, n, toPtr((*n)-1), &cone, afac.Off(0, 1), ldafac, c.Off(0, 1), ldc)
		} else {
			goblas.Ztrmm(Right, mat.UploByte(uplo), Trans, Unit, n, toPtr((*n)-1), &cone, afac.Off(1, 0), ldafac, c.Off(0, 1), ldc)
		}
	}

	//     Apply symmetric pivots
	for j = (*n); j >= 1; j-- {
		i = (*ipiv)[j-1]
		if i != j {
			goblas.Zswap(n, c.CVector(j-1, 0), ldc, c.CVector(i-1, 0), ldc)
		}
	}
	for j = (*n); j >= 1; j-- {
		i = (*ipiv)[j-1]
		if i != j {
			goblas.Zswap(n, c.CVector(0, j-1), func() *int { y := 1; return &y }(), c.CVector(0, i-1), func() *int { y := 1; return &y }())
		}
	}

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
