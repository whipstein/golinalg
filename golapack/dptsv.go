package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// DPTSV computes the solution to a real system of linear equations
// A*X = B, where A is an N-by-N symmetric positive definite tridiagonal
// matrix, and X and B are N-by-NRHS matrices.
//
// A is factored as A = L*D*L**T, and the factored form of A is then
// used to solve the system of equations.
func Dptsv(n, nrhs *int, d, e *mat.Vector, b *mat.Matrix, ldb *int, info *int) {
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
		gltest.Xerbla([]byte("DPTSV "), -(*info))
		return
	}

	//     Compute the L*D*L**T (or U**T*D*U) factorization of A.
	Dpttrf(n, d, e, info)
	if (*info) == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		Dpttrs(n, nrhs, d, e, b, ldb, info)
	}
}
