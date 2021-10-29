package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// DsysvRk computes the solution to a real system of linear
// equations A * X = B, where A is an N-by-N symmetric matrix
// and X and B are N-by-NRHS matrices.
//
// The bounded Bunch-Kaufman (rook) diagonal pivoting method is used
// to factor A as
//    A = P*U*D*(U**T)*(P**T),  if UPLO = 'U', or
//    A = P*L*D*(L**T)*(P**T),  if UPLO = 'L',
// where U (or L) is unit upper (or lower) triangular matrix,
// U**T (or L**T) is the transpose of U (or L), P is a permutation
// matrix, P**T is the transpose of P, and D is symmetric and block
// diagonal with 1-by-1 and 2-by-2 diagonal blocks.
//
// DSYTRF_RK is called to compute the factorization of a real
// symmetric matrix.  The factored form of A is then used to solve
// the system of equations A * X = B by calling BLAS3 routine DSYTRS_3.
func DsysvRk(uplo mat.MatUplo, n, nrhs int, a *mat.Matrix, e *mat.Vector, ipiv *[]int, b *mat.Matrix, work *mat.Vector, lwork int) (info int, err error) {
	var lquery bool
	var lwkopt int

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
	} else if lwork < 1 && !lquery {
		err = fmt.Errorf("lwork < 1 && !lquery: lwork=%v, lquery=%v", lwork, lquery)
	}

	if err == nil {
		if n == 0 {
			lwkopt = 1
		} else {
			if info, err = DsytrfRk(uplo, n, a, e, ipiv, work, -1); err != nil {
				panic(err)
			}
			lwkopt = int(work.Get(0))
		}
		work.Set(0, float64(lwkopt))
	}

	if err != nil || info != 0 {
		gltest.Xerbla2("DsysvRk", err)
		return
	} else if lquery {
		return
	}

	//     Compute the factorization A = P*U*D*(U**T)*(P**T) or
	//     A = P*U*D*(U**T)*(P**T).
	if info, err = DsytrfRk(uplo, n, a, e, ipiv, work, lwork); err != nil {
		panic(err)
	}

	if info == 0 {
		//        Solve the system A*X = B with BLAS3 solver, overwriting B with X.
		if info, err = Dsytrs3(uplo, n, nrhs, a, e, ipiv, b); err != nil {
			panic(err)
		}

	}

	work.Set(0, float64(lwkopt))

	return
}
