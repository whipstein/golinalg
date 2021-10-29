package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zhet01Rook reconstructs a complex Hermitian indefinite matrix A from its
// block L*D*L' or U*D*U' factorization and computes the residual
//    norm( C - A ) / ( N * norm(A) * EPS ),
// where C is the reconstructed matrix, EPS is the machine epsilon,
// L' is the transpose of L, and U' is the transpose of U.
func zhet01Rook(uplo mat.MatUplo, n int, a, afac *mat.CMatrix, ipiv *[]int, c *mat.CMatrix, rwork *mat.Vector) (resid float64) {
	var cone, czero complex128
	var anorm, eps, one, zero float64
	var i, j int
	var err error

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Quick exit if N = 0.
	if n <= 0 {
		resid = zero
		return
	}

	//     Determine EPS and the norm of A.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Zlanhe('1', uplo, n, a, rwork)

	//     Check the imaginary parts of the diagonal elements and return with
	//     an error code if any are nonzero.
	for j = 1; j <= n; j++ {
		if afac.GetIm(j-1, j-1) != zero {
			resid = one / eps
			return
		}
	}

	//     Initialize C to the identity matrix.
	golapack.Zlaset(Full, n, n, czero, cone, c)

	//     Call ZLAVHE_ROOK to form the product D * U' (or D * L' ).
	if err = zlavheRook(uplo, ConjTrans, NonUnit, n, n, afac, ipiv, c); err != nil {
		panic(err)
	}

	//     Call ZLAVHE_ROOK again to multiply by U (or L ).
	if err = zlavheRook(uplo, NoTrans, Unit, n, n, afac, ipiv, c); err != nil {
		panic(err)
	}

	//     Compute the difference  C - A .
	if uplo == Upper {
		for j = 1; j <= n; j++ {
			for i = 1; i <= j-1; i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-a.Get(i-1, j-1))
			}
			c.Set(j-1, j-1, c.Get(j-1, j-1)-a.GetReCmplx(j-1, j-1))
		}
	} else {
		for j = 1; j <= n; j++ {
			c.Set(j-1, j-1, c.Get(j-1, j-1)-a.GetReCmplx(j-1, j-1))
			for i = j + 1; i <= n; i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-a.Get(i-1, j-1))
			}
		}
	}

	//     Compute norm( C - A ) / ( N * norm(A) * EPS )
	resid = golapack.Zlanhe('1', uplo, n, c, rwork)

	if anorm <= zero {
		if resid != zero {
			resid = one / eps
		}
	} else {
		resid = ((resid / float64(n)) / anorm) / eps
	}

	return
}
