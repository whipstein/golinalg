package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dpbt01 reconstructs a symmetric positive definite band matrix A from
// its L*L' or U'*U factorization and computes the residual
//    norm( L*L' - A ) / ( N * norm(A) * EPS ) or
//    norm( U'*U - A ) / ( N * norm(A) * EPS ),
// where EPS is the machine epsilon, L' is the conjugate transpose of
// L, and U' is the conjugate transpose of U.
func dpbt01(uplo mat.MatUplo, n, kd int, a, afac *mat.Matrix, rwork *mat.Vector) (resid float64) {
	var anorm, eps, one, t, zero float64
	var i, j, k, kc, klen, ml, mu int
	var err error

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0.
	if n <= 0 {
		resid = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Dlansb('1', uplo, n, kd, a, rwork)
	if anorm <= zero {
		resid = one / eps
		return
	}

	//     Compute the product U'*U, overwriting U.
	if uplo == Upper {
		for k = n; k >= 1; k-- {
			kc = max(1, kd+2-k)
			klen = kd + 1 - kc

			//           Compute the (K,K) element of the result.
			t = goblas.Ddot(klen+1, afac.Vector(kc-1, k-1, 1), afac.Vector(kc-1, k-1, 1))
			afac.Set(kd, k-1, t)

			//           Compute the rest of column K.
			if klen > 0 {
				if err = goblas.Dtrmv(mat.Upper, mat.Trans, mat.NonUnit, klen, afac.Off(kd, k-klen-1).UpdateRows(afac.Rows-1), afac.Vector(kc-1, k-1, 1)); err != nil {
					panic(err)
				}
			}

		}

		//     UPLO = 'L':  Compute the product L*L', overwriting L.
	} else {
		for k = n; k >= 1; k-- {
			klen = min(kd, n-k)

			//           Add a multiple of column K of the factor L to each of
			//           columns K+1 through N.
			if klen > 0 {
				if err = goblas.Dsyr(mat.Lower, klen, one, afac.Vector(1, k-1, 1), afac.Off(0, k).UpdateRows(afac.Rows-1)); err != nil {
					panic(err)
				}
			}

			//           Scale column K by the diagonal element.
			t = afac.Get(0, k-1)
			goblas.Dscal(klen+1, t, afac.Vector(0, k-1, 1))

		}
	}

	//     Compute the difference  L*L' - A  or  U'*U - A.
	if uplo == Upper {
		for j = 1; j <= n; j++ {
			mu = max(1, kd+2-j)
			for i = mu; i <= kd+1; i++ {
				afac.Set(i-1, j-1, afac.Get(i-1, j-1)-a.Get(i-1, j-1))
			}
		}
	} else {
		for j = 1; j <= n; j++ {
			ml = min(kd+1, n-j+1)
			for i = 1; i <= ml; i++ {
				afac.Set(i-1, j-1, afac.Get(i-1, j-1)-a.Get(i-1, j-1))
			}
		}
	}

	//     Compute norm( L*L' - A ) / ( N * norm(A) * EPS )
	resid = golapack.Dlansb('I', uplo, n, kd, afac, rwork)

	resid = ((resid / float64(n)) / anorm) / eps

	return
}
