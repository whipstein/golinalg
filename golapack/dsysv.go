package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// DSYSV computes the solution to a real system of linear equations
//    A * X = B,
// where A is an N-by-N symmetric matrix and X and B are N-by-NRHS
// matrices.
//
// The diagonal pivoting method is used to factor A as
//    A = U * D * U**T,  if UPLO = 'U', or
//    A = L * D * L**T,  if UPLO = 'L',
// where U (or L) is a product of permutation and unit upper (lower)
// triangular matrices, and D is symmetric and block diagonal with
// 1-by-1 and 2-by-2 diagonal blocks.  The factored form of A is then
// used to solve the system of equations A * X = B.
func Dsysv(uplo byte, n, nrhs *int, a *mat.Matrix, lda *int, ipiv *[]int, b *mat.Matrix, ldb *int, work *mat.Vector, lwork, info *int) {
	var lquery bool
	var lwkopt int

	//     Test the input parameters.
	(*info) = 0
	lquery = ((*lwork) == -1)
	if uplo != 'U' && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*lda) < max(1, *n) {
		(*info) = -5
	} else if (*ldb) < max(1, *n) {
		(*info) = -8
	} else if (*lwork) < 1 && !lquery {
		(*info) = -10
	}
	//
	if (*info) == 0 {
		if (*n) == 0 {
			lwkopt = 1
		} else {
			Dsytrf(uplo, n, a, lda, ipiv, work, toPtr(-1), info)
			lwkopt = int(work.Get(0))
		}
		work.Set(0, float64(lwkopt))
	}
	//
	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYSV "), -(*info))
		return
	} else if lquery {
		return
	}

	//     Compute the factorization A = U*D*U**T or A = L*D*L**T.
	Dsytrf(uplo, n, a, lda, ipiv, work, lwork, info)
	if (*info) == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		if (*lwork) < (*n) {
			//        Solve with TRS ( Use Level BLAS 2)
			Dsytrs(uplo, n, nrhs, a, lda, ipiv, b, ldb, info)

		} else {
			//        Solve with TRS2 ( Use Level BLAS 3)
			Dsytrs2(uplo, n, nrhs, a, lda, ipiv, b, ldb, work, info)

		}

	}

	work.Set(0, float64(lwkopt))
}
