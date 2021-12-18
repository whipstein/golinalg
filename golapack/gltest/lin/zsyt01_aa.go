package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zsyt01Aa reconstructs a hermitian indefinite matrix A from its
// block L*D*L' or U*D*U' factorization and computes the residual
//    norm( C - A ) / ( N * norm(A) * EPS ),
// where C is the reconstructed matrix and EPS is the machine epsilon.
func zsyt01Aa(uplo mat.MatUplo, n int, a, afac *mat.CMatrix, ipiv *[]int, c *mat.CMatrix, rwork *mat.Vector) (resid float64) {
	var cone, czero complex128
	var anorm, eps, one, zero float64
	var i, j int
	var err error

	zero = 0.0
	one = 1.0
	czero = 0.0
	cone = 1.0

	//     Quick exit if N = 0.
	if n <= 0 {
		resid = zero
		return
	}

	//     Determine EPS and the norm of A.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Zlansy('1', uplo, n, a, rwork)

	//     Initialize C to the tridiagonal matrix T.
	golapack.Zlaset(Full, n, n, czero, czero, c)
	golapack.Zlacpy(Full, 1, n, afac.Off(0, 0).UpdateRows(afac.Rows+1), c.Off(0, 0).UpdateRows(c.Rows+1))
	if n > 1 {
		if uplo == Upper {
			golapack.Zlacpy(Full, 1, n-1, afac.Off(0, 1).UpdateRows(afac.Rows+1), c.Off(0, 1).UpdateRows(c.Rows+1))
			golapack.Zlacpy(Full, 1, n-1, afac.Off(0, 1).UpdateRows(afac.Rows+1), c.Off(1, 0).UpdateRows(c.Rows+1))
		} else {
			golapack.Zlacpy(Full, 1, n-1, afac.Off(1, 0).UpdateRows(afac.Rows+1), c.Off(0, 1).UpdateRows(c.Rows+1))
			golapack.Zlacpy(Full, 1, n-1, afac.Off(1, 0).UpdateRows(afac.Rows+1), c.Off(1, 0).UpdateRows(c.Rows+1))
		}

		//        Call ZTRMM to form the product U' * D (or L * D ).
		if uplo == Upper {
			if err = c.Off(1, 0).Trmm(Left, uplo, Trans, Unit, n-1, n, cone, afac.Off(0, 1)); err != nil {
				panic(err)
			}
		} else {
			if err = c.Off(1, 0).Trmm(Left, uplo, NoTrans, Unit, n-1, n, cone, afac.Off(1, 0)); err != nil {
				panic(err)
			}
		}

		//        Call ZTRMM again to multiply by U (or L ).
		if uplo == Upper {
			if err = c.Off(0, 1).Trmm(Right, uplo, NoTrans, Unit, n, n-1, cone, afac.Off(0, 1)); err != nil {
				panic(err)
			}
		} else {
			if err = c.Off(0, 1).Trmm(Right, uplo, Trans, Unit, n, n-1, cone, afac.Off(1, 0)); err != nil {
				panic(err)
			}
		}
	}

	//     Apply symmetric pivots
	for j = n; j >= 1; j-- {
		i = (*ipiv)[j-1]
		if i != j {
			c.Off(i-1, 0).CVector().Swap(n, c.Off(j-1, 0).CVector(), c.Rows, c.Rows)
		}
	}
	for j = n; j >= 1; j-- {
		i = (*ipiv)[j-1]
		if i != j {
			c.Off(0, i-1).CVector().Swap(n, c.Off(0, j-1).CVector(), 1, 1)
		}
	}

	//     Compute the difference  C - A .
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

	//     Compute norm( C - A ) / ( N * norm(A) * EPS )
	resid = golapack.Zlansy('1', uplo, n, c, rwork)

	if anorm <= zero {
		if resid != zero {
			resid = one / eps
		}
	} else {
		resid = ((resid / float64(n)) / anorm) / eps
	}

	return
}
