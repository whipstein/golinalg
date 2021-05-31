package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgeqrt computes a blocked QR factorization of a complex M-by-N matrix A
// using the compact WY representation of Q.
func Zgeqrt(m, n, nb *int, a *mat.CMatrix, lda *int, t *mat.CMatrix, ldt *int, work *mat.CVector, info *int) {
	var useRecursiveQr bool
	var i, ib, iinfo, k int

	useRecursiveQr = true

	//     Test the input arguments
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nb) < 1 || ((*nb) > minint(*m, *n) && minint(*m, *n) > 0) {
		(*info) = -3
	} else if (*lda) < maxint(1, *m) {
		(*info) = -5
	} else if (*ldt) < (*nb) {
		(*info) = -7
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGEQRT"), -(*info))
		return
	}

	//     Quick return if possible
	k = minint(*m, *n)
	if k == 0 {
		return
	}

	//     Blocked loop of length K
	for i = 1; i <= k; i += (*nb) {
		ib = minint(k-i+1, *nb)

		//     Compute the QR factorization of the current block A(I:M,I:I+IB-1)
		if useRecursiveQr {
			Zgeqrt3(toPtr((*m)-i+1), &ib, a.Off(i-1, i-1), lda, t.Off(0, i-1), ldt, &iinfo)
		} else {
			Zgeqrt2(toPtr((*m)-i+1), &ib, a.Off(i-1, i-1), lda, t.Off(0, i-1), ldt, &iinfo)
		}
		if i+ib <= (*n) {
			//     Update by applying H**H to A(I:M,I+IB:N) from the left
			Zlarfb('L', 'C', 'F', 'C', toPtr((*m)-i+1), toPtr((*n)-i-ib+1), &ib, a.Off(i-1, i-1), lda, t.Off(0, i-1), ldt, a.Off(i-1, i+ib-1), lda, work.CMatrix((*n)-i-ib+1, opts), toPtr((*n)-i-ib+1))
		}
	}
}
