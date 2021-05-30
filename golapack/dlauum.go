package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dlauum computes the product U * U**T or L**T * L, where the triangular
// factor U or L is stored in the upper or lower triangular part of
// the array A.
//
// If UPLO = 'U' or 'u' then the upper triangle of the result is stored,
// overwriting the factor U in A.
// If UPLO = 'L' or 'l' then the lower triangle of the result is stored,
// overwriting the factor L in A.
//
// This is the blocked form of the algorithm, calling Level 3 BLAS.
func Dlauum(uplo byte, n *int, a *mat.Matrix, lda, info *int) {
	var upper bool
	var one float64
	var i, ib, nb int

	one = 1.0

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *n) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLAUUM"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Determine the block size for this environment.
	nb = Ilaenv(toPtr(1), []byte("DLAUUM"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))

	if nb <= 1 || nb >= (*n) {
		//        Use unblocked code
		Dlauu2(uplo, n, a, lda, info)
	} else {
		//        Use blocked code
		if upper {
			//           Compute the product U * U**T.
			for i = 1; i <= (*n); i += nb {
				ib = minint(nb, (*n)-i+1)
				goblas.Dtrmm(mat.Right, mat.Upper, mat.Trans, mat.NonUnit, toPtr(i-1), &ib, &one, a.Off(i-1, i-1), lda, a.Off(0, i-1), lda)
				Dlauu2('U', &ib, a.Off(i-1, i-1), lda, info)
				if i+ib <= (*n) {
					goblas.Dgemm(mat.NoTrans, mat.Trans, toPtr(i-1), &ib, toPtr((*n)-i-ib+1), &one, a.Off(0, i+ib-1), lda, a.Off(i-1, i+ib-1), lda, &one, a.Off(0, i-1), lda)
					goblas.Dsyrk(mat.Upper, mat.NoTrans, &ib, toPtr((*n)-i-ib+1), &one, a.Off(i-1, i+ib-1), lda, &one, a.Off(i-1, i-1), lda)
				}
			}
		} else {
			//           Compute the product L**T * L.
			for i = 1; i <= (*n); i += nb {
				ib = minint(nb, (*n)-i+1)
				goblas.Dtrmm(mat.Left, mat.Lower, mat.Trans, mat.NonUnit, &ib, toPtr(i-1), &one, a.Off(i-1, i-1), lda, a.Off(i-1, 0), lda)
				Dlauu2('L', &ib, a.Off(i-1, i-1), lda, info)
				if i+ib <= (*n) {
					goblas.Dgemm(mat.Trans, mat.NoTrans, &ib, toPtr(i-1), toPtr((*n)-i-ib+1), &one, a.Off(i+ib-1, i-1), lda, a.Off(i+ib-1, 0), lda, &one, a.Off(i-1, 0), lda)
					goblas.Dsyrk(mat.Lower, mat.Trans, &ib, toPtr((*n)-i-ib+1), &one, a.Off(i+ib-1, i-1), lda, &one, a.Off(i-1, i-1), lda)
				}
			}
		}
	}
}
