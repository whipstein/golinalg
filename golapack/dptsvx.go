package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// DPTSVX uses the factorization A = L*D*L**T to compute the solution
// to a real system of linear equations A*X = B, where A is an N-by-N
// symmetric positive definite tridiagonal matrix and X and B are
// N-by-NRHS matrices.
//
// Error bounds on the solution and a condition estimate are also
// provided.
func Dptsvx(fact byte, n, nrhs *int, d, e, df, ef *mat.Vector, b *mat.Matrix, ldb *int, x *mat.Matrix, ldx *int, rcond *float64, ferr, berr, work *mat.Vector, info *int) {
	var nofact bool
	var anorm, zero float64

	zero = 0.0

	//     Test the input parameters.
	(*info) = 0
	nofact = fact == 'N'
	if !nofact && fact != 'F' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*ldb) < max(1, *n) {
		(*info) = -9
	} else if (*ldx) < max(1, *n) {
		(*info) = -11
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DPTSVX"), -(*info))
		return
	}

	if nofact {
		//        Compute the L*D*L**T (or U**T*D*U) factorization of A.
		goblas.Dcopy(*n, d.Off(0, 1), df.Off(0, 1))
		if (*n) > 1 {
			goblas.Dcopy((*n)-1, e.Off(0, 1), ef.Off(0, 1))
		}
		Dpttrf(n, df, ef, info)

		//        Return if INFO is non-zero.
		if (*info) > 0 {
			(*rcond) = zero
			return
		}
	}

	//     Compute the norm of the matrix A.
	anorm = Dlanst('1', n, d, e)

	//     Compute the reciprocal of the condition number of A.
	Dptcon(n, df, ef, &anorm, rcond, work, info)

	//     Compute the solution vectors X.
	Dlacpy('F', n, nrhs, b, ldb, x, ldx)
	Dpttrs(n, nrhs, df, ef, x, ldx, info)

	//     Use iterative refinement to improve the computed solutions and
	//     compute error bounds and backward error estimates for them.
	Dptrfs(n, nrhs, d, e, df, ef, b, ldb, x, ldx, ferr, berr, work, info)

	//     Set INFO = N+1 if the matrix is singular to working precision.
	if (*rcond) < Dlamch(Epsilon) {
		(*info) = (*n) + 1
	}
}
