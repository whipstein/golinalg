package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgelqt computes a blocked LQ factorization of a complex M-by-N matrix A
// using the compact WY representation of Q.
func Zgelqt(m, n, mb *int, a *mat.CMatrix, lda *int, t *mat.CMatrix, ldt *int, work *mat.CVector, info *int) {
	var i, ib, iinfo, k int

	//     Test the input arguments
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*mb) < 1 || ((*mb) > min(*m, *n) && min(*m, *n) > 0) {
		(*info) = -3
	} else if (*lda) < max(1, *m) {
		(*info) = -5
	} else if (*ldt) < (*mb) {
		(*info) = -7
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGELQT"), -(*info))
		return
	}

	//     Quick return if possible
	k = min(*m, *n)
	if k == 0 {
		return
	}

	//     Blocked loop of length K
	for i = 1; i <= k; i += (*mb) {
		ib = min(k-i+1, *mb)

		//     Compute the LQ factorization of the current block A(I:M,I:I+IB-1)
		Zgelqt3(&ib, toPtr((*n)-i+1), a.Off(i-1, i-1), lda, t.Off(0, i-1), ldt, &iinfo)
		if i+ib <= (*m) {
			//     Update by applying H**T to A(I:M,I+IB:N) from the right
			Zlarfb('R', 'N', 'F', 'R', toPtr((*m)-i-ib+1), toPtr((*n)-i+1), &ib, a.Off(i-1, i-1), lda, t.Off(0, i-1), ldt, a.Off(i+ib-1, i-1), lda, work.CMatrix((*m)-i-ib+1, opts), toPtr((*m)-i-ib+1))
		}
	}
}
