package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// DsysvAa2stage computes the solution to a real system of
// linear equations
//    A * X = B,
// where A is an N-by-N symmetric matrix and X and B are N-by-NRHS
// matrices.
//
// Aasen's 2-stage algorithm is used to factor A as
//    A = U**T * T * U,  if UPLO = 'U', or
//    A = L * T * L**T,  if UPLO = 'L',
// where U (or L) is a product of permutation and unit upper (lower)
// triangular matrices, and T is symmetric and band. The matrix T is
// then LU-factored with partial pivoting. The factored form of A
// is then used to solve the system of equations A * X = B.
//
// This is the blocked version of the algorithm, calling Level 3 BLAS.
func DsysvAa2stage(uplo mat.MatUplo, n, nrhs int, a, tb *mat.Matrix, ltb int, ipiv *[]int, ipiv2 *[]int, b *mat.Matrix, work *mat.Vector, lwork int) (info int, err error) {
	var tquery, upper, wquery bool
	var lwkopt int

	//     Test the input parameters.
	upper = uplo == Upper
	wquery = (lwork == -1)
	tquery = (ltb == -1)
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if ltb < (4*n) && !tquery {
		err = fmt.Errorf("ltb < (4*n) && !tquery: ltb=%v, n=%v, tquery=%v", ltb, n, tquery)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if lwork < n && !wquery {
		err = fmt.Errorf("lwork < n && !wquery: lwork=%v, n=%v, wquery=%v", lwork, n, wquery)
	}

	if err == nil {
		if info, err = DsytrfAa2stage(uplo, n, a, tb.VectorIdx(0), -1, ipiv, ipiv2, work, -1); err != nil {
			panic(err)
		}
		lwkopt = int(work.Get(0))
	}

	if err != nil || info != 0 {
		gltest.Xerbla2("DsysvAa2stage", err)
		return
	} else if wquery || tquery {
		return
	}

	//     Compute the factorization A = U**T*T*U or A = L*T*L**T.
	if info, err = DsytrfAa2stage(uplo, n, a, tb.VectorIdx(0), ltb, ipiv, ipiv2, work, lwork); err != nil {
		panic(err)
	}
	if info == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		if info, err = DsytrsAa2stage(uplo, n, nrhs, a, tb.VectorIdx(0), ltb, ipiv, ipiv2, b); err != nil {
			panic(err)
		}

	}

	work.Set(0, float64(lwkopt))

	return
}
