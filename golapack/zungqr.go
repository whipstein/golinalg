package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zungqr generates an M-by-N complex matrix Q with orthonormal columns,
// which is defined as the first N columns of a product of K elementary
// reflectors of order M
//
//       Q  =  H(1) H(2) . . . H(k)
//
// as returned by ZGEQRF.
func Zungqr(m, n, k *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var zero complex128
	var i, ib, iinfo, iws, j, ki, kk, l, ldwork, lwkopt, nb, nbmin, nx int

	zero = (0.0 + 0.0*1i)

	//     Test the input arguments
	(*info) = 0
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNGQR"), []byte{' '}, m, n, k, toPtr(-1))
	lwkopt = maxint(1, *n) * nb
	work.SetRe(0, float64(lwkopt))
	lquery = ((*lwork) == -1)
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 || (*n) > (*m) {
		(*info) = -2
	} else if (*k) < 0 || (*k) > (*n) {
		(*info) = -3
	} else if (*lda) < maxint(1, *m) {
		(*info) = -5
	} else if (*lwork) < maxint(1, *n) && !lquery {
		(*info) = -8
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNGQR"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) <= 0 {
		work.Set(0, 1)
		return
	}

	nbmin = 2
	nx = 0
	iws = (*n)
	if nb > 1 && nb < (*k) {
		//        Determine when to cross over from blocked to unblocked code.
		nx = maxint(0, Ilaenv(func() *int { y := 3; return &y }(), []byte("ZUNGQR"), []byte{' '}, m, n, k, toPtr(-1)))
		if nx < (*k) {
			//           Determine if workspace is large enough for blocked code.
			ldwork = (*n)
			iws = ldwork * nb
			if (*lwork) < iws {
				//              Not enough workspace to use optimal NB:  reduce NB and
				//              determine the minimum value of NB.
				nb = (*lwork) / ldwork
				nbmin = maxint(2, Ilaenv(func() *int { y := 2; return &y }(), []byte("ZUNGQR"), []byte{' '}, m, n, k, toPtr(-1)))
			}
		}
	}

	if nb >= nbmin && nb < (*k) && nx < (*k) {
		//        Use blocked code after the last block.
		//        The first kk columns are handled by the block method.
		ki = (((*k) - nx - 1) / nb) * nb
		kk = minint(*k, ki+nb)

		//        Set A(1:kk,kk+1:n) to zero.
		for j = kk + 1; j <= (*n); j++ {
			for i = 1; i <= kk; i++ {
				a.Set(i-1, j-1, zero)
			}
		}
	} else {
		kk = 0
	}

	//     Use unblocked code for the last or only block.
	if kk < (*n) {
		Zung2r(toPtr((*m)-kk), toPtr((*n)-kk), toPtr((*k)-kk), a.Off(kk+1-1, kk+1-1), lda, tau.Off(kk+1-1), work, &iinfo)
	}

	if kk > 0 {
		//        Use blocked code
		for i = ki + 1; i >= 1; i -= nb {
			ib = minint(nb, (*k)-i+1)
			if i+ib <= (*n) {
				//              Form the triangular factor of the block reflector
				//              H = H(i) H(i+1) . . . H(i+ib-1)
				Zlarft('F', 'C', toPtr((*m)-i+1), &ib, a.Off(i-1, i-1), lda, tau.Off(i-1), work.CMatrix(ldwork, opts), &ldwork)

				//              Apply H to A(i:m,i+ib:n) from the left
				Zlarfb('L', 'N', 'F', 'C', toPtr((*m)-i+1), toPtr((*n)-i-ib+1), &ib, a.Off(i-1, i-1), lda, work.CMatrix(ldwork, opts), &ldwork, a.Off(i-1, i+ib-1), lda, work.CMatrixOff(ib+1-1, ldwork, opts), &ldwork)
			}

			//           Apply H to rows i:m of current block
			Zung2r(toPtr((*m)-i+1), &ib, &ib, a.Off(i-1, i-1), lda, tau.Off(i-1), work, &iinfo)

			//           Set rows 1:i-1 of current block to zero
			for j = i; j <= i+ib-1; j++ {
				for l = 1; l <= i-1; l++ {
					a.Set(l-1, j-1, zero)
				}
			}
		}
	}

	work.SetRe(0, float64(iws))
}
