package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgesv computes the solution to a complex system of linear equations
//    A * X = B,
// where A is an N-by-N matrix and X and B are N-by-NRHS matrices.
//
// The LU decomposition with partial pivoting and row interchanges is
// used to factor A as
//    A = P * L * U,
// where P is a permutation matrix, L is unit lower triangular, and U is
// upper triangular.  The factored form of A is then used to solve the
// system of equations A * X = B.
func Zgesv(n, nrhs int, a *mat.CMatrix, ipiv *[]int, b *mat.CMatrix) (info int, err error) {
	//     Test the input parameters.
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zgesv", err)
		return
	}

	//     Compute the LU factorization of A.
	if info, err = Zgetrf(n, n, a, ipiv); err != nil {
		panic(err)
	}
	if info == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		if err = Zgetrs(NoTrans, n, nrhs, a, ipiv, b); err != nil {
			panic(err)
		}
	}

	return
}
