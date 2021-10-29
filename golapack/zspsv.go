package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zspsv computes the solution to a complex system of linear equations
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
func Zspsv(uplo mat.MatUplo, n, nrhs int, ap *mat.CVector, ipiv *[]int, b *mat.CMatrix) (info int, err error) {
	//     Test the input parameters.
	if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zspsv", err)
		return
	}

	//     Compute the factorization A = U*D*U**T or A = L*D*L**T.
	if info, err = Zsptrf(uplo, n, ap, ipiv); err != nil {
		panic(err)
	}
	if info == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		if err = Zsptrs(uplo, n, nrhs, ap, ipiv, b); err != nil {
			panic(err)
		}

	}

	return
}
