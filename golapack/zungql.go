package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zungql generates an M-by-N complex matrix Q with orthonormal columns,
// which is defined as the last N columns of a product of K elementary
// reflectors of order M
//
//       Q  =  H(k) . . . H(2) H(1)
//
// as returned by ZGEQLF.
func Zungql(m, n, k *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var zero complex128
	var i, ib, iinfo, iws, j, kk, l, ldwork, lwkopt, nb, nbmin, nx int

	zero = (0.0 + 0.0*1i)

	//     Test the input arguments
	(*info) = 0
	lquery = ((*lwork) == -1)
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 || (*n) > (*m) {
		(*info) = -2
	} else if (*k) < 0 || (*k) > (*n) {
		(*info) = -3
	} else if (*lda) < maxint(1, *m) {
		(*info) = -5
	}

	if (*info) == 0 {
		if (*n) == 0 {
			lwkopt = 1
		} else {
			nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNGQL"), []byte{' '}, m, n, k, toPtr(-1))
			lwkopt = (*n) * nb
		}
		work.SetRe(0, float64(lwkopt))

		if (*lwork) < maxint(1, *n) && !lquery {
			(*info) = -8
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNGQL"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) <= 0 {
		return
	}

	nbmin = 2
	nx = 0
	iws = (*n)
	if nb > 1 && nb < (*k) {
		//        Determine when to cross over from blocked to unblocked code.
		nx = maxint(0, Ilaenv(func() *int { y := 3; return &y }(), []byte("ZUNGQL"), []byte{' '}, m, n, k, toPtr(-1)))
		if nx < (*k) {
			//           Determine if workspace is large enough for blocked code.
			ldwork = (*n)
			iws = ldwork * nb
			if (*lwork) < iws {
				//              Not enough workspace to use optimal NB:  reduce NB and
				//              determine the minimum value of NB.
				nb = (*lwork) / ldwork
				nbmin = maxint(2, Ilaenv(func() *int { y := 2; return &y }(), []byte("ZUNGQL"), []byte{' '}, m, n, k, toPtr(-1)))
			}
		}
	}

	if nb >= nbmin && nb < (*k) && nx < (*k) {
		//        Use blocked code after the first block.
		//        The last kk columns are handled by the block method.
		kk = minint(*k, (((*k)-nx+nb-1)/nb)*nb)

		//        Set A(m-kk+1:m,1:n-kk) to zero.
		for j = 1; j <= (*n)-kk; j++ {
			for i = (*m) - kk + 1; i <= (*m); i++ {
				a.Set(i-1, j-1, zero)
			}
		}
	} else {
		kk = 0
	}

	//     Use unblocked code for the first or only block.
	Zung2l(toPtr((*m)-kk), toPtr((*n)-kk), toPtr((*k)-kk), a, lda, tau, work, &iinfo)

	if kk > 0 {
		//        Use blocked code
		for i = (*k) - kk + 1; i <= (*k); i += nb {
			ib = minint(nb, (*k)-i+1)
			if (*n)-(*k)+i > 1 {
				//              Form the triangular factor of the block reflector
				//              H = H(i+ib-1) . . . H(i+1) H(i)
				Zlarft('B', 'C', toPtr((*m)-(*k)+i+ib-1), &ib, a.Off(0, (*n)-(*k)+i-1), lda, tau.Off(i-1), work.CMatrix(ldwork, opts), &ldwork)

				//              Apply H to A(1:m-k+i+ib-1,1:n-k+i-1) from the left
				Zlarfb('L', 'N', 'B', 'C', toPtr((*m)-(*k)+i+ib-1), toPtr((*n)-(*k)+i-1), &ib, a.Off(0, (*n)-(*k)+i-1), lda, work.CMatrix(ldwork, opts), &ldwork, a, lda, work.CMatrixOff(ib+1-1, ldwork, opts), &ldwork)
			}

			//           Apply H to rows 1:m-k+i+ib-1 of current block
			Zung2l(toPtr((*m)-(*k)+i+ib-1), &ib, &ib, a.Off(0, (*n)-(*k)+i-1), lda, tau.Off(i-1), work, &iinfo)

			//           Set rows m-k+i+ib:m of current block to zero
			for j = (*n) - (*k) + i; j <= (*n)-(*k)+i+ib-1; j++ {
				for l = (*m) - (*k) + i + ib; l <= (*m); l++ {
					a.Set(l-1, j-1, zero)
				}
			}
		}
	}

	work.SetRe(0, float64(iws))
}
