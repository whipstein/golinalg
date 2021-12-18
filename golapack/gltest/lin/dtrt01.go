package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dtrt01 computes the residual for a triangular matrix A times its
// inverse:
//    RESID = norm( A*AINV - I ) / ( N * norm(A) * norm(AINV) * EPS ),
// where EPS is the machine epsilon.
func dtrt01(uplo mat.MatUplo, diag mat.MatDiag, n int, a, ainv *mat.Matrix, work *mat.Vector) (rcond, resid float64) {
	var ainvnm, anorm, eps, one, zero float64
	var j int
	var err error

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0
	if n <= 0 {
		rcond = one
		resid = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0 or AINVNM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Dlantr('1', uplo, diag, n, n, a, work)
	ainvnm = golapack.Dlantr('1', uplo, diag, n, n, ainv, work)
	if anorm <= zero || ainvnm <= zero {
		rcond = zero
		resid = one / eps
		return
	}
	rcond = (one / anorm) / ainvnm

	//     Set the diagonal of AINV to 1 if AINV has unit diagonal.
	if diag == Unit {
		for j = 1; j <= n; j++ {
			ainv.Set(j-1, j-1, one)
		}
	}

	//     Compute A * AINV, overwriting AINV.
	if uplo == Upper {
		for j = 1; j <= n; j++ {
			if err = ainv.Off(0, j-1).Vector().Trmv(Upper, NoTrans, diag, j, a, 1); err != nil {
				panic(err)
			}
		}
	} else {
		for j = 1; j <= n; j++ {
			if err = ainv.Off(j-1, j-1).Vector().Trmv(Lower, NoTrans, diag, n-j+1, a.Off(j-1, j-1), 1); err != nil {
				panic(err)
			}
		}
	}

	//     Subtract 1 from each diagonal element to form A*AINV - I.
	for j = 1; j <= n; j++ {
		ainv.Set(j-1, j-1, ainv.Get(j-1, j-1)-one)
	}

	//     Compute norm(A*AINV - I) / (N * norm(A) * norm(AINV) * EPS)
	resid = golapack.Dlantr('1', uplo, NonUnit, n, n, ainv, work)

	resid = ((resid * rcond) / float64(n)) / eps

	return
}
