package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zsytri2 computes the inverse of a COMPLEX*16 symmetric indefinite matrix
// A using the factorization A = U*D*U**T or A = L*D*L**T computed by
// ZSYTRF. ZSYTRI2 sets the LEADING DIMENSION of the workspace
// before calling ZSYTRI2X that actually computes the inverse.
func Zsytri2(uplo byte, n *int, a *mat.CMatrix, lda *int, ipiv *[]int, work *mat.CVector, lwork, info *int) {
	var lquery, upper bool
	var minsize, nbmax int

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	lquery = ((*lwork) == -1)
	//     Get blocksize
	nbmax = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZSYTRI2"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))
	if nbmax >= (*n) {
		minsize = (*n)
	} else {
		minsize = ((*n) + nbmax + 1) * (nbmax + 3)
	}

	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *n) {
		(*info) = -4
	} else if (*lwork) < minsize && !lquery {
		(*info) = -7
	}

	//     Quick return if possible
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZSYTRI2"), -(*info))
		return
	} else if lquery {
		work.SetRe(0, float64(minsize))
		return
	}
	if (*n) == 0 {
		return
	}
	if nbmax >= (*n) {
		Zsytri(uplo, n, a, lda, ipiv, work, info)
	} else {
		Zsytri2x(uplo, n, a, lda, ipiv, work.CMatrix(*n+nbmax+1, opts), &nbmax, info)
	}
}
