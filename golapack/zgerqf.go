package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgerqf computes an RQ factorization of a complex M-by-N matrix A:
// A = R * Q.
func Zgerqf(m, n *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var i, ib, iinfo, iws, k, ki, kk, ldwork, lwkopt, mu, nb, nbmin, nu, nx int

	//     Test the input arguments
	(*info) = 0
	lquery = ((*lwork) == -1)
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *m) {
		(*info) = -4
	}

	if (*info) == 0 {
		k = min(*m, *n)
		if k == 0 {
			lwkopt = 1
		} else {
			nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGERQF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
			lwkopt = (*m) * nb
		}
		work.SetRe(0, float64(lwkopt))

		if (*lwork) < max(1, *m) && !lquery {
			(*info) = -7
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGERQF"), -(*info))
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
	iws = (*m)
	if nb > 1 && nb < k {
		//        Determine when to cross over from blocked to unblocked code.
		nx = max(0, Ilaenv(func() *int { y := 3; return &y }(), []byte("ZGERQF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1)))
		if nx < k {
			//           Determine if workspace is large enough for blocked code.
			ldwork = (*m)
			iws = ldwork * nb
			if (*lwork) < iws {
				//              Not enough workspace to use optimal NB:  reduce NB and
				//              determine the minimum value of NB.
				nb = (*lwork) / ldwork
				nbmin = max(2, Ilaenv(func() *int { y := 2; return &y }(), []byte("ZGERQF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1)))
			}
		}
	}

	if nb >= nbmin && nb < k && nx < k {
		//        Use blocked code initially.
		//        The last kk rows are handled by the block method.
		ki = ((k - nx - 1) / nb) * nb
		kk = min(k, ki+nb)

		for i = k - kk + ki + 1; i >= k-kk+1; i -= nb {
			ib = min(k-i+1, nb)

			//           Compute the RQ factorization of the current block
			//           A(m-k+i:m-k+i+ib-1,1:n-k+i+ib-1)
			Zgerq2(&ib, toPtr((*n)-k+i+ib-1), a.Off((*m)-k+i-1, 0), lda, tau.Off(i-1), work, &iinfo)
			if (*m)-k+i > 1 {
				//              Form the triangular factor of the block reflector
				//              H = H(i+ib-1) . . . H(i+1) H(i)
				Zlarft('B', 'R', toPtr((*n)-k+i+ib-1), &ib, a.Off((*m)-k+i-1, 0), lda, tau.Off(i-1), work.CMatrix(ldwork, opts), &ldwork)

				//              Apply H to A(1:m-k+i-1,1:n-k+i+ib-1) from the right
				Zlarfb('R', 'N', 'B', 'R', toPtr((*m)-k+i-1), toPtr((*n)-k+i+ib-1), &ib, a.Off((*m)-k+i-1, 0), lda, work.CMatrix(ldwork, opts), &ldwork, a, lda, work.CMatrixOff(ib, ldwork, opts), &ldwork)
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
		Zgerq2(&mu, &nu, a, lda, tau, work, &iinfo)
	}

	work.SetRe(0, float64(iws))
}
