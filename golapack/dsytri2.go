package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsytri2 computes the inverse of a DOUBLE PRECISION symmetric indefinite matrix
// A using the factorization A = U*D*U**T or A = L*D*L**T computed by
// DSYTRF. DSYTRI2 sets the LEADING DIMENSION of the workspace
// before calling DSYTRI2X that actually computes the inverse.
func Dsytri2(uplo byte, n *int, a *mat.Matrix, lda *int, ipiv *[]int, work *mat.Matrix, lwork *int, info *int) {
	var lquery, upper bool
	var minsize, nbmax int

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	lquery = ((*lwork) == -1)
	//     Get blocksize
	nbmax = Ilaenv(func() *int { y := 1; return &y }(), []byte("DSYTRI2"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))
	if nbmax >= (*n) {
		minsize = (*n)
	} else {
		minsize = ((*n) + nbmax + 1) * (nbmax + 3)
	}
	//
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *n) {
		(*info) = -4
	} else if (*lwork) < minsize && !lquery {
		(*info) = -7
	}

	//     Quick return if possible
	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYTRI2"), -(*info))
		return
	} else if lquery {
		work.SetIdx(0, float64(minsize))
		return
	}
	if (*n) == 0 {
		return
	}
	if nbmax >= (*n) {
		Dsytri(uplo, n, a, lda, ipiv, work.VectorIdx(0), info)
	} else {
		Dsytri2x(uplo, n, a, lda, ipiv, work.VectorIdx(0), &nbmax, info)
	}
}
