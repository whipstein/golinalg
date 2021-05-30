package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dsytri3 computes the inverse of a real symmetric indefinite
// matrix A using the factorization computed by DSYTRF_RK or DSYTRF_BK:
//
//     A = P*U*D*(U**T)*(P**T) or A = P*L*D*(L**T)*(P**T),
//
// where U (or L) is unit upper (or lower) triangular matrix,
// U**T (or L**T) is the transpose of U (or L), P is a permutation
// matrix, P**T is the transpose of P, and D is symmetric and block
// diagonal with 1-by-1 and 2-by-2 diagonal blocks.
//
// DSYTRI_3 sets the leading dimension of the workspace  before calling
// DSYTRI_3X that actually computes the inverse.  This is the blocked
// version of the algorithm, calling Level 3 BLAS.
func Dsytri3(uplo byte, n *int, a *mat.Matrix, lda *int, e *mat.Vector, ipiv *[]int, work *mat.Vector, lwork *int, info *int) {
	var lquery, upper bool
	var lwkopt, nb int

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	lquery = ((*lwork) == -1)

	//     Determine the block size
	nb = maxint(1, Ilaenv(func() *int { y := 1; return &y }(), []byte("DSYTRI_3"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1)))
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
	//
	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYTRI_3"), -(*info))
		return
	} else if lquery {
		work.Set(0, float64(lwkopt))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	Dsytri3x(uplo, n, a, lda, e, ipiv, work, &nb, info)

	work.Set(0, float64(lwkopt))
}
