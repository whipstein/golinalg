package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgbsv computes the solution to a complex system of linear equations
// A * X = B, where A is a band matrix of order N with KL subdiagonals
// and KU superdiagonals, and X and B are N-by-NRHS matrices.
//
// The LU decomposition with partial pivoting and row interchanges is
// used to factor A as A = L * U, where L is a product of permutation
// and unit lower triangular matrices with KL subdiagonals, and U is
// upper triangular with KL+KU superdiagonals.  The factored form of A
// is then used to solve the system of equations A * X = B.
func Zgbsv(n, kl, ku, nrhs int, ab *mat.CMatrix, ipiv *[]int, b *mat.CMatrix) (info int, err error) {
	//     Test the input parameters.
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if kl < 0 {
		err = fmt.Errorf("kl < 0: kl=%v", kl)
	} else if ku < 0 {
		err = fmt.Errorf("ku < 0: ku=%v", ku)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if ab.Rows < 2*kl+ku+1 {
		err = fmt.Errorf("ab.Rows < 2*kl+ku+1: ab.Rows=%v, kl=%v, ku=%v", ab.Rows, kl, ku)
	} else if b.Rows < max(n, 1) {
		err = fmt.Errorf("b.Rows < max(n, 1): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zgbsv", err)
		return
	}

	//     Compute the LU factorization of the band matrix A.
	if info, err = Zgbtrf(n, n, kl, ku, ab, ipiv); err != nil {
		panic(err)
	}
	if info == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		if err = Zgbtrs(NoTrans, n, kl, ku, nrhs, ab, ipiv, b); err != nil {
			panic(err)
		}
	}

	return
}
