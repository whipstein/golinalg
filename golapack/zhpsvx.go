package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhpsvx uses the diagonal pivoting factorization A = U*D*U**H or
// A = L*D*L**H to compute the solution to a complex system of linear
// equations A * X = B, where A is an N-by-N Hermitian matrix stored
// in packed format and X and B are N-by-NRHS matrices.
//
// Error bounds on the solution and a condition estimate are also
// provided.
func Zhpsvx(fact, uplo byte, n, nrhs *int, ap, afp *mat.CVector, ipiv *[]int, b *mat.CMatrix, ldb *int, x *mat.CMatrix, ldx *int, rcond *float64, ferr, berr *mat.Vector, work *mat.CVector, rwork *mat.Vector, info *int) {
	var nofact bool
	var anorm, zero float64

	zero = 0.0

	//     Test the input parameters.
	(*info) = 0
	nofact = fact == 'N'
	if !nofact && fact != 'F' {
		(*info) = -1
	} else if uplo != 'U' && uplo != 'L' {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*nrhs) < 0 {
		(*info) = -4
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -9
	} else if (*ldx) < maxint(1, *n) {
		(*info) = -11
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHPSVX"), -(*info))
		return
	}

	if nofact {
		//        Compute the factorization A = U*D*U**H or A = L*D*L**H.
		goblas.Zcopy(toPtr((*n)*((*n)+1)/2), ap, func() *int { y := 1; return &y }(), afp, func() *int { y := 1; return &y }())
		Zhptrf(uplo, n, afp, ipiv, info)

		//        Return if INFO is non-zero.
		if (*info) > 0 {
			(*rcond) = zero
			return
		}
	}

	//     Compute the norm of the matrix A.
	anorm = Zlanhp('I', uplo, n, ap, rwork)

	//     Compute the reciprocal of the condition number of A.
	Zhpcon(uplo, n, afp, ipiv, &anorm, rcond, work, info)

	//     Compute the solution vectors X.
	Zlacpy('F', n, nrhs, b, ldb, x, ldx)
	Zhptrs(uplo, n, nrhs, afp, ipiv, x, ldx, info)

	//     Use iterative refinement to improve the computed solutions and
	//     compute error bounds and backward error estimates for them.
	Zhprfs(uplo, n, nrhs, ap, afp, ipiv, b, ldb, x, ldx, ferr, berr, work, rwork, info)

	//     Set INFO = N+1 if the matrix is singular to working precision.
	if (*rcond) < Dlamch(Epsilon) {
		(*info) = (*n) + 1
	}
}
