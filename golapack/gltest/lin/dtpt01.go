package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dtpt01 computes the residual for a triangular matrix A times its
// inverse when A is stored in packed format:
//    RESID = norm(A*AINV - I) / ( N * norm(A) * norm(AINV) * EPS ),
// where EPS is the machine epsilon.
func dtpt01(uplo mat.MatUplo, diag mat.MatDiag, n int, ap, ainvp *mat.Vector, work *mat.Vector) (rcond, resid float64) {
	var unitd bool
	var ainvnm, anorm, eps, one, zero float64
	var j, jc int
	var err error

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0.
	if n <= 0 {
		rcond = one
		resid = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0 or AINVNM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Dlantp('1', uplo, diag, n, ap, work)
	ainvnm = golapack.Dlantp('1', uplo, diag, n, ainvp, work)
	if anorm <= zero || ainvnm <= zero {
		rcond = zero
		resid = one / eps
		return
	}
	rcond = (one / anorm) / ainvnm

	//     Compute A * AINV, overwriting AINV.
	unitd = diag == Unit
	if uplo == Upper {
		jc = 1
		for j = 1; j <= n; j++ {
			if unitd {
				ainvp.Set(jc+j-1-1, one)
			}

			//           Form the j-th column of A*AINV
			if err = goblas.Dtpmv(mat.Upper, mat.NoTrans, diag, j, ap, ainvp.Off(jc-1, 1)); err != nil {
				panic(err)
			}

			//           Subtract 1 from the diagonal
			ainvp.Set(jc+j-1-1, ainvp.Get(jc+j-1-1)-one)
			jc = jc + j
		}
	} else {
		jc = 1
		for j = 1; j <= n; j++ {
			if unitd {
				ainvp.Set(jc-1, one)
			}

			//           Form the j-th column of A*AINV
			if err = goblas.Dtpmv(Lower, NoTrans, diag, n-j+1, ap.Off(jc-1), ainvp.Off(jc-1, 1)); err != nil {
				panic(err)
			}

			//           Subtract 1 from the diagonal
			ainvp.Set(jc-1, ainvp.Get(jc-1)-one)
			jc = jc + n - j + 1
		}
	}

	//     Compute norm(A*AINV - I) / (N * norm(A) * norm(AINV) * EPS)
	resid = golapack.Dlantp('1', uplo, NonUnit, n, ainvp, work)

	resid = ((resid * rcond) / float64(n)) / eps

	return
}
