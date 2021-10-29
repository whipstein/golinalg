package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dptsv computes the solution to a real system of linear equations
// A*X = B, where A is an N-by-N symmetric positive definite tridiagonal
// matrix, and X and B are N-by-NRHS matrices.
//
// A is factored as A = L*D*L**T, and the factored form of A is then
// used to solve the system of equations.
func Dptsv(n, nrhs int, d, e *mat.Vector, b *mat.Matrix) (info int, err error) {
	//     Test the input parameters.
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dptsv", err)
		return
	}

	//     Compute the L*D*L**T (or U**T*D*U) factorization of A.
	if info, err = Dpttrf(n, d, e); err != nil {
		panic(err)
	}
	if info == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		if err = Dpttrs(n, nrhs, d, e, b); err != nil {
			panic(err)
		}
	}

	return
}
