package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dspsv computes the solution to a real system of linear equations
//    A * X = B,
// where A is an N-by-N symmetric matrix stored in packed format and X
// and B are N-by-NRHS matrices.
//
// The diagonal pivoting method is used to factor A as
//    A = U * D * U**T,  if UPLO = 'U', or
//    A = L * D * L**T,  if UPLO = 'L',
// where U (or L) is a product of permutation and unit upper (lower)
// triangular matrices, D is symmetric and block diagonal with 1-by-1
// and 2-by-2 diagonal blocks.  The factored form of A is then used to
// solve the system of equations A * X = B.
func Dspsv(uplo byte, n, nrhs *int, ap *mat.Vector, ipiv *[]int, b *mat.Matrix, ldb, info *int) {
	//     Test the input parameters.
	(*info) = 0
	if uplo != 'U' && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -7
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DSPSV "), -(*info))
		return
	}

	//     Compute the factorization A = U*D*U**T or A = L*D*L**T.
	Dsptrf(uplo, n, ap, ipiv, info)
	if (*info) == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		Dsptrs(uplo, n, nrhs, ap, ipiv, b, ldb, info)

	}
}
