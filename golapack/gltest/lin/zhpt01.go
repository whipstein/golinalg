package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zhpt01 reconstructs a Hermitian indefinite packed matrix A from its
// block L*D*L' or U*D*U' factorization and computes the residual
//    norm( C - A ) / ( N * norm(A) * EPS ),
// where C is the reconstructed matrix, EPS is the machine epsilon,
// L' is the conjugate transpose of L, and U' is the conjugate transpose
// of U.
func Zhpt01(uplo byte, n *int, a, afac *mat.CVector, ipiv *[]int, c *mat.CMatrix, ldc *int, rwork *mat.Vector, resid *float64) {
	var cone, czero complex128
	var anorm, eps, one, zero float64
	var i, info, j, jc int

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
	anorm = golapack.Zlanhp('1', uplo, n, a, rwork)

	//     Check the imaginary parts of the diagonal elements and return with
	//     an error code if any are nonzero.
	jc = 1
	if uplo == 'U' {
		for j = 1; j <= (*n); j++ {
			if afac.GetIm(jc-1) != zero {
				(*resid) = one / eps
				return
			}
			jc = jc + j + 1
		}
	} else {
		for j = 1; j <= (*n); j++ {
			if afac.GetIm(jc-1) != zero {
				(*resid) = one / eps
				return
			}
			jc = jc + (*n) - j + 1
		}
	}

	//     Initialize C to the identity matrix.
	golapack.Zlaset('F', n, n, &czero, &cone, c, ldc)

	//     Call ZLAVHP to form the product D * U' (or D * L' ).
	Zlavhp(uplo, 'C', 'N', n, n, afac, ipiv, c, ldc, &info)

	//     Call ZLAVHP again to multiply by U ( or L ).
	Zlavhp(uplo, 'N', 'U', n, n, afac, ipiv, c, ldc, &info)

	//     Compute the difference  C - A .
	if uplo == 'U' {
		jc = 0
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= j-1; i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-a.Get(jc+i-1))
			}
			c.Set(j-1, j-1, c.Get(j-1, j-1)-a.GetReCmplx(jc+j-1))
			jc = jc + j
		}
	} else {
		jc = 1
		for j = 1; j <= (*n); j++ {
			c.Set(j-1, j-1, c.Get(j-1, j-1)-a.GetReCmplx(jc-1))
			for i = j + 1; i <= (*n); i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-a.Get(jc+i-j-1))
			}
			jc = jc + (*n) - j + 1
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
