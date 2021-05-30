package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dgeqlf computes a QL factorization of a real M-by-N matrix A:
// A = Q * L.
func Dgeqlf(m, n *int, a *mat.Matrix, lda *int, tau, work *mat.Vector, lwork, info *int) {
	var lquery bool
	var i, ib, iinfo, iws, k, ki, kk, ldwork, lwkopt, mu, nb, nbmin, nu, nx int

	//     Test the input arguments
	(*info) = 0
	lquery = ((*lwork) == -1)
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *m) {
		(*info) = -4
	}
	//
	if (*info) == 0 {
		k = minint(*m, *n)
		if k == 0 {
			lwkopt = 1
		} else {
			nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DGEQLF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
			lwkopt = (*n) * nb
		}
		work.Set(0, float64(lwkopt))

		if (*lwork) < maxint(1, *n) && !lquery {
			(*info) = -7
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGEQLF"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if k == 0 {
		return
	}

	nbmin = 2
	nx = 1
	iws = (*n)
	if nb > 1 && nb < k {
		//        Determine when to cross over from blocked to unblocked code.
		nx = maxint(0, Ilaenv(func() *int { y := 3; return &y }(), []byte("DGEQLF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1)))
		if nx < k {
			//           Determine if workspace is large enough for blocked code.
			ldwork = (*n)
			iws = ldwork * nb
			if (*lwork) < iws {
				//              Not enough workspace to use optimal NB:  reduce NB and
				//              determine the minimum value of NB.
				nb = (*lwork) / ldwork
				nbmin = maxint(2, Ilaenv(func() *int { y := 2; return &y }(), []byte("DGEQLF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1)))
			}
		}
	}

	if nb >= nbmin && nb < k && nx < k {
		//        Use blocked code initially.
		//        The last kk columns are handled by the block method.
		ki = ((k - nx - 1) / nb) * nb
		kk = minint(k, ki+nb)

		for i = k - kk + ki + 1; i >= k-kk+1; i -= nb {
			ib = minint(k-i+1, nb)

			//           Compute the QL factorization of the current block
			//           A(1:m-k+i+ib-1,n-k+i:n-k+i+ib-1)
			Dgeql2(toPtr((*m)-k+i+ib-1), &ib, a.Off(0, (*n)-k+i-1), lda, tau.Off(i-1), work, &iinfo)
			if (*n)-k+i > 1 {
				//              Form the triangular factor of the block reflector
				//              H = H(i+ib-1) . . . H(i+1) H(i)
				Dlarft('B', 'C', toPtr((*m)-k+i+ib-1), &ib, a.Off(0, (*n)-k+i-1), lda, tau.Off(i-1), work.Matrix(ldwork, opts), &ldwork)

				//              Apply H**T to A(1:m-k+i+ib-1,1:n-k+i-1) from the left
				Dlarfb('L', 'T', 'B', 'C', toPtr((*m)-k+i+ib-1), toPtr((*n)-k+i-1), &ib, a.Off(0, (*n)-k+i-1), lda, work.Matrix(ldwork, opts), &ldwork, a, lda, work.MatrixOff(ib+1-1, ldwork, opts), &ldwork)
			}
		}
		mu = (*m) - k + i + nb - 1
		nu = (*n) - k + i + nb - 1
	} else {
		mu = (*m)
		nu = (*n)
	}

	//     Use unblocked code to factor the last or only block
	if mu > 0 && nu > 0 {
		Dgeql2(&mu, &nu, a, lda, tau, work, &iinfo)
	}

	work.Set(0, float64(iws))
}
