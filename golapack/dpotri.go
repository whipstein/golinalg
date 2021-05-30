package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dpotri computes the inverse of a real symmetric positive definite
// matrix A using the Cholesky factorization A = U**T*U or A = L*L**T
// computed by DPOTRF.
func Dpotri(uplo byte, n *int, a *mat.Matrix, lda, info *int) {
	//     Test the input parameters.
	(*info) = 0
	if uplo != 'U' && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *n) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DPOTRI"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Invert the triangular Cholesky factor U or L.
	Dtrtri(uplo, 'N', n, a, lda, info)
	if (*info) > 0 {
		return
	}

	//     Form inv(U) * inv(U)**T or inv(L)**T * inv(L).
	Dlauum(uplo, n, a, lda, info)
}
