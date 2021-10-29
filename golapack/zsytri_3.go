package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsytri3 computes the inverse of a complex symmetric indefinite
// matrix A using the factorization computed by ZSYTRF_RK or ZSYTRF_BK:
//
//     A = P*U*D*(U**T)*(P**T) or A = P*L*D*(L**T)*(P**T),
//
// where U (or L) is unit upper (or lower) triangular matrix,
// U**T (or L**T) is the transpose of U (or L), P is a permutation
// matrix, P**T is the transpose of P, and D is symmetric and block
// diagonal with 1-by-1 and 2-by-2 diagonal blocks.
//
// Zsytri3 sets the leading dimension of the workspace  before calling
// Zsytri3X that actually computes the inverse.  This is the blocked
// version of the algorithm, calling Level 3 BLAS.
func Zsytri3(uplo mat.MatUplo, n int, a *mat.CMatrix, e *mat.CVector, ipiv *[]int, work *mat.CVector, lwork int) (info int, err error) {
	var lquery, upper bool
	var lwkopt, nb int

	//     Test the input parameters.
	upper = uplo == Upper
	lquery = (lwork == -1)

	//     Determine the block size
	nb = max(1, Ilaenv(1, "Zsytri3", []byte{uplo.Byte()}, n, -1, -1, -1))
	lwkopt = (n + nb + 1) * (nb + 3)

	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if lwork < lwkopt && !lquery {
		err = fmt.Errorf("lwork < lwkopt && !lquery: lwork=%v, lwkopt=%v, lquery=%v", lwork, lwkopt, lquery)
	}

	if err != nil {
		gltest.Xerbla2("Zsytri3", err)
		return
	} else if lquery {
		work.SetRe(0, float64(lwkopt))
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if info, err = Zsytri3x(uplo, n, a, e, ipiv, work.CMatrix(n+nb+1, opts), nb); err != nil {
		panic(err)
	}

	work.SetRe(0, float64(lwkopt))

	return
}
