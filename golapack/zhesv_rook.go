package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zhesvrook computes the solution to a complex system of linear equations
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
// ZHETRF_ROOK is called to compute the factorization of a complex
// Hermition matrix A using the bounded Bunch-Kaufman ("rook") diagonal
// pivoting method.
//
// The factored form of A is then used to solve the system
// of equations A * X = B by calling ZHETRS_ROOK (uses BLAS 2).
func Zhesvrook(uplo byte, n, nrhs *int, a *mat.CMatrix, lda *int, ipiv *[]int, b *mat.CMatrix, ldb *int, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var lwkopt, nb int

	//     Test the input parameters.
	(*info) = 0
	lquery = ((*lwork) == -1)
	if uplo != 'U' && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -8
	} else if (*lwork) < 1 && !lquery {
		(*info) = -10
	}
	//
	if (*info) == 0 {
		if (*n) == 0 {
			lwkopt = 1
		} else {
			nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZHETRF_ROOK"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))
			lwkopt = (*n) * nb
		}
		work.SetRe(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHESV_ROOK"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Compute the factorization A = U*D*U**H or A = L*D*L**H.
	Zhetrfrook(uplo, n, a, lda, ipiv, work, lwork, info)
	if (*info) == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		//
		//        Solve with TRS ( Use Level BLAS 2)
		Zhetrsrook(uplo, n, nrhs, a, lda, ipiv, b, ldb, info)

	}

	work.SetRe(0, float64(lwkopt))
}
