package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zpotri computes the inverse of a complex Hermitian positive definite
// matrix A using the Cholesky factorization A = U**H*U or A = L*L**H
// computed by ZPOTRF.
func Zpotri(uplo byte, n *int, a *mat.CMatrix, lda, info *int) {
	//     Test the input parameters.
	(*info) = 0
	if uplo != 'U' && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *n) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZPOTRI"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Invert the triangular Cholesky factor U or L.
	Ztrtri(uplo, 'N', n, a, lda, info)
	if (*info) > 0 {
		return
	}

	//     Form inv(U) * inv(U)**H or inv(L)**H * inv(L).
	Zlauum(uplo, n, a, lda, info)
}
