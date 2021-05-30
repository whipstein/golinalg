package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zptsv computes the solution to a complex system of linear equations
// A*X = B, where A is an N-by-N Hermitian positive definite tridiagonal
// matrix, and X and B are N-by-NRHS matrices.
//
// A is factored as A = L*D*L**H, and the factored form of A is then
// used to solve the system of equations.
func Zptsv(n, nrhs *int, d *mat.Vector, e *mat.CVector, b *mat.CMatrix, ldb, info *int) {
	//     Test the input parameters.
	(*info) = 0
	if (*n) < 0 {
		(*info) = -1
	} else if (*nrhs) < 0 {
		(*info) = -2
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -6
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZPTSV "), -(*info))
		return
	}

	//     Compute the L*D*L**H (or U**H*D*U) factorization of A.
	Zpttrf(n, d, e, info)
	if (*info) == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		Zpttrs('L', n, nrhs, d, e, b, ldb, info)
	}
}
