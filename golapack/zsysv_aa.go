package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// ZsysvAa computes the solution to a complex system of linear equations
//    A * X = B,
// where A is an N-by-N symmetric matrix and X and B are N-by-NRHS
// matrices.
//
// Aasen's algorithm is used to factor A as
//    A = U**T * T * U,  if UPLO = 'U', or
//    A = L * T * L**T,  if UPLO = 'L',
// where U (or L) is a product of permutation and unit upper (lower)
// triangular matrices, and T is symmetric tridiagonal. The factored
// form of A is then used to solve the system of equations A * X = B.
func ZsysvAa(uplo mat.MatUplo, n, nrhs int, a *mat.CMatrix, ipiv *[]int, b *mat.CMatrix, work *mat.CVector, lwork int) (info int, err error) {
	var lquery bool
	var lwkopt, lwkoptSytrf, lwkoptSytrs int

	//     Test the input parameters.
	lquery = (lwork == -1)
	if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if lwork < max(2*n, 3*n-2) && !lquery {
		err = fmt.Errorf("lwork < max(2*n, 3*n-2) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
	}
	//
	if err == nil {
		if err = ZsytrfAa(uplo, n, a, ipiv, work, -1); err != nil {
			panic(err)
		}
		lwkoptSytrf = int(work.GetRe(0))
		if info, err = ZsytrsAa(uplo, n, nrhs, a, ipiv, b, work, -1); err != nil {
			panic(err)
		}
		lwkoptSytrs = int(work.GetRe(0))
		lwkopt = max(lwkoptSytrf, lwkoptSytrs)
		work.SetRe(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("ZsysvAa", err)
		return
	} else if lquery {
		return
	}

	//     Compute the factorization A = U**T*T*U or A = L*T*L**T.
	if err = ZsytrfAa(uplo, n, a, ipiv, work, lwork); err != nil {
		panic(err)
	}
	if info == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		if info, err = ZsytrsAa(uplo, n, nrhs, a, ipiv, b, work, lwork); err != nil {
			panic(err)
		}

	}

	work.SetRe(0, float64(lwkopt))

	return
}
