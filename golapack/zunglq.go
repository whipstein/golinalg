package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zunglq generates an M-by-N complex matrix Q with orthonormal rows,
// which is defined as the first M rows of a product of K elementary
// reflectors of order N
//
//       Q  =  H(k)**H . . . H(2)**H H(1)**H
//
// as returned by ZGELQF.
func Zunglq(m, n, k *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var zero complex128
	var i, ib, iinfo, iws, j, ki, kk, l, ldwork, lwkopt, nb, nbmin, nx int

	zero = (0.0 + 0.0*1i)

	//     Test the input arguments
	(*info) = 0
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNGLQ"), []byte{' '}, m, n, k, toPtr(-1))
	lwkopt = max(1, *m) * nb
	work.SetRe(0, float64(lwkopt))
	lquery = ((*lwork) == -1)
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < (*m) {
		(*info) = -2
	} else if (*k) < 0 || (*k) > (*m) {
		(*info) = -3
	} else if (*lda) < max(1, *m) {
		(*info) = -5
	} else if (*lwork) < max(1, *m) && !lquery {
		(*info) = -8
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNGLQ"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*m) <= 0 {
		work.Set(0, 1)
		return
	}

	nbmin = 2
	nx = 0
	iws = (*m)
	if nb > 1 && nb < (*k) {
		//        Determine when to cross over from blocked to unblocked code.
		nx = max(0, Ilaenv(func() *int { y := 3; return &y }(), []byte("ZUNGLQ"), []byte{' '}, m, n, k, toPtr(-1)))
		if nx < (*k) {
			//           Determine if workspace is large enough for blocked code.
			ldwork = (*m)
			iws = ldwork * nb
			if (*lwork) < iws {
				//              Not enough workspace to use optimal NB:  reduce NB and
				//              determine the minimum value of NB.
				nb = (*lwork) / ldwork
				nbmin = max(2, Ilaenv(func() *int { y := 2; return &y }(), []byte("ZUNGLQ"), []byte{' '}, m, n, k, toPtr(-1)))
			}
		}
	}

	if nb >= nbmin && nb < (*k) && nx < (*k) {
		//        Use blocked code after the last block.
		//        The first kk rows are handled by the block method.
		ki = (((*k) - nx - 1) / nb) * nb
		kk = min(*k, ki+nb)

		//        Set A(kk+1:m,1:kk) to zero.
		for j = 1; j <= kk; j++ {
			for i = kk + 1; i <= (*m); i++ {
				a.Set(i-1, j-1, zero)
			}
		}
	} else {
		kk = 0
	}

	//     Use unblocked code for the last or only block.
	if kk < (*m) {
		Zungl2(toPtr(((*m) - kk)), toPtr((*n)-kk), toPtr((*k)-kk), a.Off(kk, kk), lda, tau.Off(kk), work, &iinfo)
	}

	if kk > 0 {
		//        Use blocked code
		for i = ki + 1; i >= 1; i -= nb {
			ib = min(nb, (*k)-i+1)
			if i+ib <= (*m) {
				//              Form the triangular factor of the block reflector
				//              H = H(i) H(i+1) . . . H(i+ib-1)
				Zlarft('F', 'R', toPtr((*n)-i+1), &ib, a.Off(i-1, i-1), lda, tau.Off(i-1), work.CMatrix(ldwork, opts), &ldwork)

				//              Apply H**H to A(i+ib:m,i:n) from the right
				Zlarfb('R', 'C', 'F', 'R', toPtr((*m)-i-ib+1), toPtr((*n)-i+1), &ib, a.Off(i-1, i-1), lda, work.CMatrix(ldwork, opts), &ldwork, a.Off(i+ib-1, i-1), lda, work.CMatrixOff(ib, ldwork, opts), &ldwork)
			}

			//           Apply H**H to columns i:n of current block
			Zungl2(&ib, toPtr((*n)-i+1), &ib, a.Off(i-1, i-1), lda, tau.Off(i-1), work, &iinfo)

			//           Set columns 1:i-1 of current block to zero
			for j = 1; j <= i-1; j++ {
				for l = i; l <= i+ib-1; l++ {
					a.Set(l-1, j-1, zero)
				}
			}
		}
	}

	work.SetRe(0, float64(iws))
}
