package lin

import (
	"golinalg/golapack"
	"golinalg/mat"
)

// Zspt01 reconstructs a symmetric indefinite packed matrix A from its
// diagonal pivoting factorization A = U*D*U' or A = L*D*L' and computes
// the residual
//    norm( C - A ) / ( N * norm(A) * EPS ),
// where C is the reconstructed matrix and EPS is the machine epsilon.
func Zspt01(uplo byte, n *int, a, afac *mat.CVector, ipiv *[]int, c *mat.CMatrix, ldc *int, rwork *mat.Vector, resid *float64) {
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
	anorm = golapack.Zlansp('1', uplo, n, a, rwork)

	//     Initialize C to the identity matrix.
	golapack.Zlaset('F', n, n, &czero, &cone, c, ldc)

	//     Call ZLAVSP to form the product D * U' (or D * L' ).
	Zlavsp(uplo, 'T', 'N', n, n, afac, ipiv, c, ldc, &info)

	//     Call ZLAVSP again to multiply by U ( or L ).
	Zlavsp(uplo, 'N', 'U', n, n, afac, ipiv, c, ldc, &info)

	//     Compute the difference  C - A .
	if uplo == 'U' {
		jc = 0
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= j; i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-a.Get(jc+i-1))
			}
			jc = jc + j
		}
	} else {
		jc = 1
		for j = 1; j <= (*n); j++ {
			for i = j; i <= (*n); i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-a.Get(jc+i-j-1))
			}
			jc = jc + (*n) - j + 1
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
