package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlauum computes the product U * U**H or L**H * L, where the triangular
// factor U or L is stored in the upper or lower triangular part of
// the array A.
//
// If UPLO = 'U' or 'u' then the upper triangle of the result is stored,
// overwriting the factor U in A.
// If UPLO = 'L' or 'l' then the lower triangle of the result is stored,
// overwriting the factor L in A.
//
// This is the blocked form of the algorithm, calling Level 3 BLAS.
func Zlauum(uplo byte, n *int, a *mat.CMatrix, lda, info *int) {
	var upper bool
	var cone complex128
	var one float64
	var i, ib, nb int
	var err error
	_ = err

	one = 1.0
	cone = (1.0 + 0.0*1i)

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
		gltest.Xerbla([]byte("ZLAUUM"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Determine the block size for this environment.
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZLAUUM"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))

	if nb <= 1 || nb >= (*n) {
		//        Use unblocked code
		Zlauu2(uplo, n, a, lda, info)
	} else {
		//        Use blocked code
		if upper {
			//           Compute the product U * U**H.
			for i = 1; i <= (*n); i += nb {
				ib = min(nb, (*n)-i+1)
				err = goblas.Ztrmm(Right, Upper, ConjTrans, NonUnit, i-1, ib, cone, a.Off(i-1, i-1), a.Off(0, i-1))
				Zlauu2('U', &ib, a.Off(i-1, i-1), lda, info)
				if i+ib <= (*n) {
					err = goblas.Zgemm(NoTrans, ConjTrans, i-1, ib, (*n)-i-ib+1, cone, a.Off(0, i+ib-1), a.Off(i-1, i+ib-1), cone, a.Off(0, i-1))
					err = goblas.Zherk(Upper, NoTrans, ib, (*n)-i-ib+1, one, a.Off(i-1, i+ib-1), one, a.Off(i-1, i-1))
				}
			}
		} else {
			//           Compute the product L**H * L.
			for i = 1; i <= (*n); i += nb {
				ib = min(nb, (*n)-i+1)
				err = goblas.Ztrmm(Left, Lower, ConjTrans, NonUnit, ib, i-1, cone, a.Off(i-1, i-1), a.Off(i-1, 0))
				Zlauu2('L', &ib, a.Off(i-1, i-1), lda, info)
				if i+ib <= (*n) {
					err = goblas.Zgemm(ConjTrans, NoTrans, ib, i-1, (*n)-i-ib+1, cone, a.Off(i+ib-1, i-1), a.Off(i+ib-1, 0), cone, a.Off(i-1, 0))
					err = goblas.Zherk(Lower, ConjTrans, ib, (*n)-i-ib+1, one, a.Off(i+ib-1, i-1), one, a.Off(i-1, i-1))
				}
			}
		}
	}
}
