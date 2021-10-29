package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zptsv computes the solution to a complex system of linear equations
// A*X = B, where A is an N-by-N Hermitian positive definite tridiagonal
// matrix, and X and B are N-by-NRHS matrices.
//
// A is factored as A = L*D*L**H, and the factored form of A is then
// used to solve the system of equations.
func Zptsv(n, nrhs int, d *mat.Vector, e *mat.CVector, b *mat.CMatrix) (info int, err error) {
	//     Test the input parameters.
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zptsv", err)
		return
	}

	//     Compute the L*D*L**H (or U**H*D*U) factorization of A.
	if info, err = Zpttrf(n, d, e); err != nil {
		panic(err)
	}
	if info == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		if err = Zpttrs(Lower, n, nrhs, d, e, b); err != nil {
			panic(err)
		}
	}

	return
}
