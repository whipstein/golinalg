package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zungrq generates an M-by-N complex matrix Q with orthonormal rows,
// which is defined as the last M rows of a product of K elementary
// reflectors of order N
//
//       Q  =  H(1)**H H(2)**H . . . H(k)**H
//
// as returned by ZGERQF.
func Zungrq(m, n, k *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var zero complex128
	var i, ib, ii, iinfo, iws, j, kk, l, ldwork, lwkopt, nb, nbmin, nx int

	zero = (0.0 + 0.0*1i)

	//     Test the input arguments
	(*info) = 0
	lquery = ((*lwork) == -1)
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < (*m) {
		(*info) = -2
	} else if (*k) < 0 || (*k) > (*m) {
		(*info) = -3
	} else if (*lda) < maxint(1, *m) {
		(*info) = -5
	}

	if (*info) == 0 {
		if (*m) <= 0 {
			lwkopt = 1
		} else {
			nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNGRQ"), []byte{' '}, m, n, k, toPtr(-1))
			lwkopt = (*m) * nb
		}
		work.SetRe(0, float64(lwkopt))

		if (*lwork) < maxint(1, *m) && !lquery {
			(*info) = -8
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNGRQ"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*m) <= 0 {
		return
	}

	nbmin = 2
	nx = 0
	iws = (*m)
	if nb > 1 && nb < (*k) {
		//        Determine when to cross over from blocked to unblocked code.
		nx = maxint(0, Ilaenv(func() *int { y := 3; return &y }(), []byte("ZUNGRQ"), []byte{' '}, m, n, k, toPtr(-1)))
		if nx < (*k) {
			//           Determine if workspace is large enough for blocked code.
			ldwork = (*m)
			iws = ldwork * nb
			if (*lwork) < iws {
				//              Not enough workspace to use optimal NB:  reduce NB and
				//              determine the minimum value of NB.
				nb = (*lwork) / ldwork
				nbmin = maxint(2, Ilaenv(func() *int { y := 2; return &y }(), []byte("ZUNGRQ"), []byte{' '}, m, n, k, toPtr(-1)))
			}
		}
	}

	if nb >= nbmin && nb < (*k) && nx < (*k) {
		//        Use blocked code after the first block.
		//        The last kk rows are handled by the block method.
		kk = minint(*k, (((*k)-nx+nb-1)/nb)*nb)

		//        Set A(1:m-kk,n-kk+1:n) to zero.
		for j = (*n) - kk + 1; j <= (*n); j++ {
			for i = 1; i <= (*m)-kk; i++ {
				a.Set(i-1, j-1, zero)
			}
		}
	} else {
		kk = 0
	}

	//     Use unblocked code for the first or only block.
	Zungr2(toPtr((*m)-kk), toPtr((*n)-kk), toPtr((*k)-kk), a, lda, tau, work, &iinfo)

	if kk > 0 {
		//        Use blocked code
		for i = (*k) - kk + 1; i <= (*k); i += nb {
			ib = minint(nb, (*k)-i+1)
			ii = (*m) - (*k) + i
			if ii > 1 {
				//              Form the triangular factor of the block reflector
				//              H = H(i+ib-1) . . . H(i+1) H(i)
				Zlarft('B', 'R', toPtr((*n)-(*k)+i+ib-1), &ib, a.Off(ii-1, 0), lda, tau.Off(i-1), work.CMatrix(ldwork, opts), &ldwork)

				//              Apply H**H to A(1:m-k+i-1,1:n-k+i+ib-1) from the right
				Zlarfb('R', 'C', 'B', 'R', toPtr(ii-1), toPtr((*n)-(*k)+i+ib-1), &ib, a.Off(ii-1, 0), lda, work.CMatrix(ldwork, opts), &ldwork, a, lda, work.CMatrixOff(ib+1-1, ldwork, opts), &ldwork)
			}

			//           Apply H**H to columns 1:n-k+i+ib-1 of current block
			Zungr2(&ib, toPtr((*n)-(*k)+i+ib-1), &ib, a.Off(ii-1, 0), lda, tau.Off(i-1), work, &iinfo)

			//           Set columns n-k+i+ib:n of current block to zero
			for l = (*n) - (*k) + i + ib; l <= (*n); l++ {
				for j = ii; j <= ii+ib-1; j++ {
					a.Set(j-1, l-1, zero)
				}
			}
		}
	}

	work.SetRe(0, float64(iws))
}
