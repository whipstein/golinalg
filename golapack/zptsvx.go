package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zptsvx uses the factorization A = L*D*L**H to compute the solution
// to a complex system of linear equations A*X = B, where A is an
// N-by-N Hermitian positive definite tridiagonal matrix and X and B
// are N-by-NRHS matrices.
//
// Error bounds on the solution and a condition estimate are also
// provided.
func Zptsvx(fact byte, n, nrhs *int, d *mat.Vector, e *mat.CVector, df *mat.Vector, ef *mat.CVector, b *mat.CMatrix, ldb *int, x *mat.CMatrix, ldx *int, rcond *float64, ferr, berr *mat.Vector, work *mat.CVector, rwork *mat.Vector, info *int) {
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
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -9
	} else if (*ldx) < maxint(1, *n) {
		(*info) = -11
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZPTSVX"), -(*info))
		return
	}

	if nofact {
		//        Compute the L*D*L**H (or U**H*D*U) factorization of A.
		goblas.Dcopy(n, d, func() *int { y := 1; return &y }(), df, func() *int { y := 1; return &y }())
		if (*n) > 1 {
			goblas.Zcopy(toPtr((*n)-1), e, func() *int { y := 1; return &y }(), ef, func() *int { y := 1; return &y }())
		}
		Zpttrf(n, df, ef, info)

		//        Return if INFO is non-zero.
		if (*info) > 0 {
			(*rcond) = zero
			return
		}
	}

	//     Compute the norm of the matrix A.
	anorm = Zlanht('1', n, d, e)

	//     Compute the reciprocal of the condition number of A.
	Zptcon(n, df, ef, &anorm, rcond, rwork, info)

	//     Compute the solution vectors X.
	Zlacpy('F', n, nrhs, b, ldb, x, ldx)
	Zpttrs('L', n, nrhs, df, ef, x, ldx, info)

	//     Use iterative refinement to improve the computed solutions and
	//     compute error bounds and backward error estimates for them.
	Zptrfs('L', n, nrhs, d, e, df, ef, b, ldb, x, ldx, ferr, berr, work, rwork, info)

	//     Set INFO = N+1 if the matrix is singular to working precision.
	if (*rcond) < Dlamch(Epsilon) {
		(*info) = (*n) + 1
	}
}