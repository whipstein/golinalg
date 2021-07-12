package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
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
	var err error
	_ = err

	one = 1.0

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *n) {
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
				ib = min(nb, (*n)-i+1)
				err = goblas.Dtrmm(mat.Right, mat.Upper, mat.Trans, mat.NonUnit, i-1, ib, one, a.Off(i-1, i-1), a.Off(0, i-1))
				Dlauu2('U', &ib, a.Off(i-1, i-1), lda, info)
				if i+ib <= (*n) {
					err = goblas.Dgemm(mat.NoTrans, mat.Trans, i-1, ib, (*n)-i-ib+1, one, a.Off(0, i+ib-1), a.Off(i-1, i+ib-1), one, a.Off(0, i-1))
					err = goblas.Dsyrk(mat.Upper, mat.NoTrans, ib, (*n)-i-ib+1, one, a.Off(i-1, i+ib-1), one, a.Off(i-1, i-1))
				}
			}
		} else {
			//           Compute the product L**T * L.
			for i = 1; i <= (*n); i += nb {
				ib = min(nb, (*n)-i+1)
				err = goblas.Dtrmm(mat.Left, mat.Lower, mat.Trans, mat.NonUnit, ib, i-1, one, a.Off(i-1, i-1), a.Off(i-1, 0))
				Dlauu2('L', &ib, a.Off(i-1, i-1), lda, info)
				if i+ib <= (*n) {
					err = goblas.Dgemm(mat.Trans, mat.NoTrans, ib, i-1, (*n)-i-ib+1, one, a.Off(i+ib-1, i-1), a.Off(i+ib-1, 0), one, a.Off(i-1, 0))
					err = goblas.Dsyrk(mat.Lower, mat.Trans, ib, (*n)-i-ib+1, one, a.Off(i+ib-1, i-1), one, a.Off(i-1, i-1))
				}
			}
		}
	}
}
