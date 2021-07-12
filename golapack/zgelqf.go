package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgelqf computes an LQ factorization of a complex M-by-N matrix A:
//
//    A = ( L 0 ) *  Q
//
// where:
//
//    Q is a N-by-N orthogonal matrix;
//    L is an lower-triangular M-by-M matrix;
//    0 is a M-by-(N-M) zero matrix, if M < N.
func Zgelqf(m, n *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var i, ib, iinfo, iws, k, ldwork, lwkopt, nb, nbmin, nx int

	//     Test the input arguments
	(*info) = 0
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGELQF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
	lwkopt = (*m) * nb
	work.SetRe(0, float64(lwkopt))
	lquery = ((*lwork) == -1)
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *m) {
		(*info) = -4
	} else if (*lwork) < max(1, *m) && !lquery {
		(*info) = -7
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGELQF"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	k = min(*m, *n)
	if k == 0 {
		work.Set(0, 1)
		return
	}

	nbmin = 2
	nx = 0
	iws = (*m)
	if nb > 1 && nb < k {
		//        Determine when to cross over from blocked to unblocked code.
		nx = max(0, Ilaenv(func() *int { y := 3; return &y }(), []byte("ZGELQF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1)))
		if nx < k {
			//           Determine if workspace is large enough for blocked code.
			ldwork = (*m)
			iws = ldwork * nb
			if (*lwork) < iws {
				//              Not enough workspace to use optimal NB:  reduce NB and
				//              determine the minimum value of NB.
				nb = (*lwork) / ldwork
				nbmin = max(2, Ilaenv(func() *int { y := 2; return &y }(), []byte("ZGELQF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1)))
			}
		}
	}

	if nb >= nbmin && nb < k && nx < k {
		//        Use blocked code initially
		for i = 1; i <= k-nx; i += nb {
			ib = min(k-i+1, nb)

			//           Compute the LQ factorization of the current block
			//           A(i:i+ib-1,i:n)
			Zgelq2(&ib, toPtr((*n)-i+1), a.Off(i-1, i-1), lda, tau.Off(i-1), work, &iinfo)
			if i+ib <= (*m) {
				//              Form the triangular factor of the block reflector
				//              H = H(i) H(i+1) . . . H(i+ib-1)
				Zlarft('F', 'R', toPtr((*n)-i+1), &ib, a.Off(i-1, i-1), lda, tau.Off(i-1), work.CMatrix(ldwork, opts), &ldwork)

				//              Apply H to A(i+ib:m,i:n) from the right
				Zlarfb('R', 'N', 'F', 'R', toPtr((*m)-i-ib+1), toPtr((*n)-i+1), &ib, a.Off(i-1, i-1), lda, work.CMatrix(ldwork, opts), &ldwork, a.Off(i+ib-1, i-1), lda, work.CMatrixOff(ib, ldwork, opts), &ldwork)
			}
		}
	} else {
		i = 1
	}

	//     Use unblocked code to factor the last or only block.
	if i <= k {
		Zgelq2(toPtr((*m)-i+1), toPtr((*n)-i+1), a.Off(i-1, i-1), lda, tau.Off(i-1), work, &iinfo)
	}

	work.SetRe(0, float64(iws))
}
