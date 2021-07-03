package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgtsvx uses the LU factorization to compute the solution to a complex
// system of linear equations A * X = B, A**T * X = B, or A**H * X = B,
// where A is a tridiagonal matrix of order N and X and B are N-by-NRHS
// matrices.
//
// Error bounds on the solution and a condition estimate are also
// provided.
func Zgtsvx(fact, trans byte, n, nrhs *int, dl, d, du, dlf, df, duf, du2 *mat.CVector, ipiv *[]int, b *mat.CMatrix, ldb *int, x *mat.CMatrix, ldx *int, rcond *float64, ferr, berr *mat.Vector, work *mat.CVector, rwork *mat.Vector, info *int) {
	var nofact, notran bool
	var norm byte
	var anorm, zero float64

	zero = 0.0

	(*info) = 0
	nofact = fact == 'N'
	notran = trans == 'N'
	if !nofact && fact != 'F' {
		(*info) = -1
	} else if !notran && trans != 'T' && trans != 'C' {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*nrhs) < 0 {
		(*info) = -4
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -14
	} else if (*ldx) < maxint(1, *n) {
		(*info) = -16
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGTSVX"), -(*info))
		return
	}

	if nofact {
		//        Compute the LU factorization of A.
		goblas.Zcopy(*n, d, 1, df, 1)
		if (*n) > 1 {
			goblas.Zcopy((*n)-1, dl, 1, dlf, 1)
			goblas.Zcopy((*n)-1, du, 1, duf, 1)
		}
		Zgttrf(n, dlf, df, duf, du2, ipiv, info)

		//        Return if INFO is non-zero.
		if (*info) > 0 {
			(*rcond) = zero
			return
		}
	}

	//     Compute the norm of the matrix A.
	if notran {
		norm = '1'
	} else {
		norm = 'I'
	}
	anorm = Zlangt(norm, n, dl, d, du)

	//     Compute the reciprocal of the condition number of A.
	Zgtcon(norm, n, dlf, df, duf, du2, ipiv, &anorm, rcond, work, info)

	//     Compute the solution vectors X.
	Zlacpy('F', n, nrhs, b, ldb, x, ldx)
	Zgttrs(trans, n, nrhs, dlf, df, duf, du2, ipiv, x, ldx, info)

	//     Use iterative refinement to improve the computed solutions and
	//     compute error bounds and backward error estimates for them.
	Zgtrfs(trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx, ferr, berr, work, rwork, info)

	//     Set INFO = N+1 if the matrix is singular to working precision.
	if (*rcond) < Dlamch(Epsilon) {
		(*info) = (*n) + 1
	}
}
