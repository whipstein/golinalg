package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// ZhesvRook computes the solution to a complex system of linear equations
//    A * X = B,
// where A is an N-by-N Hermitian matrix and X and B are N-by-NRHS
// matrices.
//
// The bounded Bunch-Kaufman ("rook") diagonal pivoting method is used
// to factor A as
//    A = U * D * U**T,  if UPLO = 'U', or
//    A = L * D * L**T,  if UPLO = 'L',
// where U (or L) is a product of permutation and unit upper (lower)
// triangular matrices, and D is Hermitian and block diagonal with
// 1-by-1 and 2-by-2 diagonal blocks.
//
// ZhetrfRook is called to compute the factorization of a complex
// Hermition matrix A using the bounded Bunch-Kaufman ("rook") diagonal
// pivoting method.
//
// The factored form of A is then used to solve the system
// of equations A * X = B by calling ZHETRS_ROOK (uses BLAS 2).
func ZhesvRook(uplo mat.MatUplo, n, nrhs int, a *mat.CMatrix, ipiv *[]int, b *mat.CMatrix, work *mat.CVector, lwork int) (info int, err error) {
	var lquery bool
	var lwkopt, nb int

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
			nb = Ilaenv(1, "ZhetrfRook", []byte{uplo.Byte()}, n, -1, -1, -1)
			lwkopt = n * nb
		}
		work.SetRe(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("ZhesvRook", err)
		return
	} else if lquery {
		return
	}

	//     Compute the factorization A = U*D*U**H or A = L*D*L**H.
	if info, err = ZhetrfRook(uplo, n, a, ipiv, work, lwork); err != nil {
		panic(err)
	}
	if info == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		//
		//        Solve with TRS ( Use Level BLAS 2)
		if err = ZhetrsRook(uplo, n, nrhs, a, ipiv, b); err != nil {
			panic(err)
		}

	}

	work.SetRe(0, float64(lwkopt))

	return
}
