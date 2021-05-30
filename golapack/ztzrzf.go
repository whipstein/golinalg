package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Ztzrzf reduces the M-by-N ( M<=N ) complex upper trapezoidal matrix A
// to upper triangular form by means of unitary transformations.
//
// The upper trapezoidal matrix A is factored as
//
//    A = ( R  0 ) * Z,
//
// where Z is an N-by-N unitary matrix and R is an M-by-M upper
// triangular matrix.
func Ztzrzf(m, n *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var zero complex128
	var i, ib, iws, ki, kk, ldwork, lwkmin, lwkopt, m1, mu, nb, nbmin, nx int

	zero = (0.0 + 0.0*1i)

	//     Test the input arguments
	(*info) = 0
	lquery = ((*lwork) == -1)
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < (*m) {
		(*info) = -2
	} else if (*lda) < maxint(1, *m) {
		(*info) = -4
	}

	if (*info) == 0 {
		if (*m) == 0 || (*m) == (*n) {
			lwkopt = 1
			lwkmin = 1
		} else {
			//           Determine the block size.
			nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGERQF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
			lwkopt = (*m) * nb
			lwkmin = maxint(1, *m)
		}
		work.SetRe(0, float64(lwkopt))

		if (*lwork) < lwkmin && !lquery {
			(*info) = -7
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTZRZF"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*m) == 0 {
		return
	} else if (*m) == (*n) {
		for i = 1; i <= (*n); i++ {
			tau.Set(i-1, zero)
		}
		return
	}

	nbmin = 2
	nx = 1
	iws = (*m)
	if nb > 1 && nb < (*m) {
		//        Determine when to cross over from blocked to unblocked code.
		nx = maxint(0, Ilaenv(func() *int { y := 3; return &y }(), []byte("ZGERQF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1)))
		if nx < (*m) {
			//           Determine if workspace is large enough for blocked code.
			ldwork = (*m)
			iws = ldwork * nb
			if (*lwork) < iws {
				//              Not enough workspace to use optimal NB:  reduce NB and
				//              determine the minimum value of NB.
				nb = (*lwork) / ldwork
				nbmin = maxint(2, Ilaenv(func() *int { y := 2; return &y }(), []byte("ZGERQF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1)))
			}
		}
	}

	if nb >= nbmin && nb < (*m) && nx < (*m) {
		//        Use blocked code initially.
		//        The last kk rows are handled by the block method.
		m1 = minint((*m)+1, *n)
		ki = (((*m) - nx - 1) / nb) * nb
		kk = minint(*m, ki+nb)

		for i = (*m) - kk + ki + 1; i >= (*m)-kk+1; i -= nb {
			ib = minint((*m)-i+1, nb)

			//           Compute the TZ factorization of the current block
			//           A(i:i+ib-1,i:n)
			Zlatrz(&ib, toPtr((*n)-i+1), toPtr((*n)-(*m)), a.Off(i-1, i-1), lda, tau.Off(i-1), work)
			if i > 1 {
				//              Form the triangular factor of the block reflector
				//              H = H(i+ib-1) . . . H(i+1) H(i)
				Zlarzt('B', 'R', toPtr((*n)-(*m)), &ib, a.Off(i-1, m1-1), lda, tau.Off(i-1), work.CMatrix(ldwork, opts), &ldwork)

				//              Apply H to A(1:i-1,i:n) from the right
				Zlarzb('R', 'N', 'B', 'R', toPtr(i-1), toPtr((*n)-i+1), &ib, toPtr((*n)-(*m)), a.Off(i-1, m1-1), lda, work.CMatrix(ldwork, opts), &ldwork, a.Off(0, i-1), lda, work.CMatrixOff(ib+1-1, ldwork, opts), &ldwork)
			}
		}
		mu = i + nb - 1
	} else {
		mu = (*m)
	}

	//     Use unblocked code to factor the last or only block
	if mu > 0 {
		Zlatrz(&mu, n, toPtr((*n)-(*m)), a, lda, tau, work)
	}

	work.SetRe(0, float64(lwkopt))
}
