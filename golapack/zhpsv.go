package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhpsv computes the solution to a complex system of linear equations
//    A * X = B,
// where A is an N-by-N Hermitian matrix stored in packed format and X
// and B are N-by-NRHS matrices.
//
// The diagonal pivoting method is used to factor A as
//    A = U * D * U**H,  if UPLO = 'U', or
//    A = L * D * L**H,  if UPLO = 'L',
// where U (or L) is a product of permutation and unit upper (lower)
// triangular matrices, D is Hermitian and block diagonal with 1-by-1
// and 2-by-2 diagonal blocks.  The factored form of A is then used to
// solve the system of equations A * X = B.
func Zhpsv(uplo byte, n, nrhs *int, ap *mat.CVector, ipiv *[]int, b *mat.CMatrix, ldb, info *int) {
	//     Test the input parameters.
	(*info) = 0
	if uplo != 'U' && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*ldb) < max(1, *n) {
		(*info) = -7
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHPSV "), -(*info))
		return
	}

	//     Compute the factorization A = U*D*U**H or A = L*D*L**H.
	Zhptrf(uplo, n, ap, ipiv, info)
	if (*info) == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		Zhptrs(uplo, n, nrhs, ap, ipiv, b, ldb, info)

	}
}
