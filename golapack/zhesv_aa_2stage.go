package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zhesvaa2stage computes the solution to a complex system of
// linear equations
//    A * X = B,
// where A is an N-by-N Hermitian matrix and X and B are N-by-NRHS
// matrices.
//
// Aasen's 2-stage algorithm is used to factor A as
//    A = U**H * T * U,  if UPLO = 'U', or
//    A = L * T * L**H,  if UPLO = 'L',
// where U (or L) is a product of permutation and unit upper (lower)
// triangular matrices, and T is Hermitian and band. The matrix T is
// then LU-factored with partial pivoting. The factored form of A
// is then used to solve the system of equations A * X = B.
//
// This is the blocked version of the algorithm, calling Level 3 BLAS.
func Zhesvaa2stage(uplo byte, n, nrhs *int, a *mat.CMatrix, lda *int, tb *mat.CVector, ltb *int, ipiv, ipiv2 *[]int, b *mat.CMatrix, ldb *int, work *mat.CVector, lwork, info *int) {
	var tquery, upper, wquery bool
	var lwkopt int

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	wquery = ((*lwork) == -1)
	tquery = ((*ltb) == -1)
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*ltb) < (4*(*n)) && !tquery {
		(*info) = -7
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -11
	} else if (*lwork) < (*n) && !wquery {
		(*info) = -13
	}

	if (*info) == 0 {
		Zhetrfaa2stage(uplo, n, a, lda, tb, toPtr(-1), ipiv, ipiv2, work, toPtr(-1), info)
		lwkopt = int(work.GetRe(0))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHESV_AA_2STAGE"), -(*info))
		return
	} else if wquery || tquery {
		return
	}

	//     Compute the factorization A = U**H*T*U or A = L*T*L**H.
	Zhetrfaa2stage(uplo, n, a, lda, tb, ltb, ipiv, ipiv2, work, lwork, info)
	if (*info) == 0 {
		//        Solve the system A*X = B, overwriting B with X.
		Zhetrsaa2stage(uplo, n, nrhs, a, lda, tb, ltb, ipiv, ipiv2, b, ldb, info)

	}

	work.SetRe(0, float64(lwkopt))
}
