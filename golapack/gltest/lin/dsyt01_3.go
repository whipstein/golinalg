package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dsyt013 reconstructs a symmetric indefinite matrix A from its
// block L*D*L' or U*D*U' factorization computed by DSYTRF_RK
// (or DSYTRF_BK) and computes the residual
//    norm( C - A ) / ( N * norm(A) * EPS ),
// where C is the reconstructed matrix and EPS is the machine epsilon.
func dsyt013(uplo mat.MatUplo, n int, a, afac *mat.Matrix, e *mat.Vector, ipiv []int, c *mat.Matrix, rwork *mat.Vector) (resid float64) {
	var anorm, eps, one, zero float64
	var i, j int
	var err error

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0.
	if n <= 0 {
		resid = zero
		return
	}

	//     a) Revert to multiplyers of L
	if err = golapack.DsyconvfRook(uplo, 'R', n, afac, e, &ipiv); err != nil {
		panic(err)
	}

	//     1) Determine EPS and the norm of A.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Dlansy('1', uplo, n, a, rwork)

	//     2) Initialize C to the identity matrix.
	golapack.Dlaset(Full, n, n, zero, one, c)

	//     3) Call DLAVSY_ROOK to form the product D * U' (or D * L' ).
	if err = dlavsyRook(uplo, Trans, NonUnit, n, n, afac, ipiv, c); err != nil {
		panic(err)
	}

	//     4) Call DLAVSY_ROOK again to multiply by U (or L ).
	if err = dlavsyRook(uplo, NoTrans, Unit, n, n, afac, ipiv, c); err != nil {
		panic(err)
	}

	//     5) Compute the difference  C - A.
	if uplo == Upper {
		for j = 1; j <= n; j++ {
			for i = 1; i <= j; i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-a.Get(i-1, j-1))
			}
		}
	} else {
		for j = 1; j <= n; j++ {
			for i = j; i <= n; i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-a.Get(i-1, j-1))
			}
		}
	}

	//     6) Compute norm( C - A ) / ( N * norm(A) * EPS )
	resid = golapack.Dlansy('1', uplo, n, c, rwork)

	if anorm <= zero {
		if resid != zero {
			resid = one / eps
		}
	} else {
		resid = ((resid / float64(n)) / anorm) / eps
	}

	//     b) Convert to factor of L (or U)
	if err = golapack.DsyconvfRook(uplo, 'C', n, afac, e, &ipiv); err != nil {
		panic(err)
	}

	return
}
