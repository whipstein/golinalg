package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zhetri3 computes the inverse of a complex Hermitian indefinite
// matrix A using the factorization computed by ZHETRF_RK or ZHETRF_BK:
//
//     A = P*U*D*(U**H)*(P**T) or A = P*L*D*(L**H)*(P**T),
//
// where U (or L) is unit upper (or lower) triangular matrix,
// U**H (or L**H) is the conjugate of U (or L), P is a permutation
// matrix, P**T is the transpose of P, and D is Hermitian and block
// diagonal with 1-by-1 and 2-by-2 diagonal blocks.
//
// ZHETRI_3 sets the leading dimension of the workspace  before calling
// ZHETRI_3X that actually computes the inverse.  This is the blocked
// version of the algorithm, calling Level 3 BLAS.
func Zhetri3(uplo byte, n *int, a *mat.CMatrix, lda *int, e *mat.CVector, ipiv *[]int, work *mat.CVector, lwork, info *int) {
	var lquery, upper bool
	var lwkopt, nb int

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	lquery = ((*lwork) == -1)

	//     Determine the block size
	nb = maxint(1, Ilaenv(func() *int { y := 1; return &y }(), []byte("ZHETRI_3"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1)))
	lwkopt = ((*n) + nb + 1) * (nb + 3)

	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *n) {
		(*info) = -4
	} else if (*lwork) < lwkopt && !lquery {
		(*info) = -8
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHETRI_3"), -(*info))
		return
	} else if lquery {
		work.SetRe(0, float64(lwkopt))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	Zhetri3x(uplo, n, a, lda, e, ipiv, work.CMatrix(*n+nb+1, opts), &nb, info)

	work.SetRe(0, float64(lwkopt))
}
