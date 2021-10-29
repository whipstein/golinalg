package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dspt01 reconstructs a symmetric indefinite packed matrix A from its
// block L*D*L' or U*D*U' factorization and computes the residual
//      norm( C - A ) / ( N * norm(A) * EPS ),
// where C is the reconstructed matrix and EPS is the machine epsilon.
func dspt01(uplo mat.MatUplo, n int, a, afac *mat.Vector, ipiv []int, c *mat.Matrix, rwork *mat.Vector) (resid float64) {
	var anorm, eps, one, zero float64
	var i, j, jc int
	var err error

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0.
	if n <= 0 {
		resid = zero
		return
	}

	//     Determine EPS and the norm of A.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Dlansp('1', uplo, n, a, rwork)

	//     Initialize C to the identity matrix.
	golapack.Dlaset(Full, n, n, zero, one, c)

	//     Call DLAVSP to form the product D * U' (or D * L' ).
	if err = dlavsp(uplo, Trans, NonUnit, n, n, afac, ipiv, c); err != nil {
		panic(err)
	}

	//     Call DLAVSP again to multiply by U ( or L ).
	if err = dlavsp(uplo, NoTrans, Unit, n, n, afac, ipiv, c); err != nil {
		panic(err)
	}

	//     Compute the difference  C - A .
	if uplo == Upper {
		jc = 0
		for j = 1; j <= n; j++ {
			for i = 1; i <= j; i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-a.Get(jc+i-1))
			}
			jc += j
		}
	} else {
		jc = 1
		for j = 1; j <= n; j++ {
			for i = j; i <= n; i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-a.Get(jc+i-j-1))
			}
			jc += n - j + 1
		}
	}

	//     Compute norm( C - A ) / ( N * norm(A) * EPS )
	resid = golapack.Dlansy('1', uplo, n, c, rwork)

	if anorm <= zero {
		if resid != zero {
			resid = one / eps
		}
	} else {
		resid = ((resid / float64(n)) / anorm) / eps
	}

	return
}
