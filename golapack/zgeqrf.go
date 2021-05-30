package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zgeqrf computes a QR factorization of a complex M-by-N matrix A:
//
//    A = Q * ( R ),
//            ( 0 )
//
// where:
//
//    Q is a M-by-M orthogonal matrix;
//    R is an upper-triangular N-by-N matrix;
//    0 is a (M-N)-by-N zero matrix, if M > N.
func Zgeqrf(m, n *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var i, ib, iinfo, iws, k, ldwork, lwkopt, nb, nbmin, nx int

	//     Test the input arguments
	(*info) = 0
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEQRF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
	lwkopt = (*n) * nb
	work.SetRe(0, float64(lwkopt))
	lquery = ((*lwork) == -1)
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *m) {
		(*info) = -4
	} else if (*lwork) < maxint(1, *n) && !lquery {
		(*info) = -7
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGEQRF"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	k = minint(*m, *n)
	if k == 0 {
		work.Set(0, 1)
		return
	}

	nbmin = 2
	nx = 0
	iws = (*n)
	if nb > 1 && nb < k {
		//        Determine when to cross over from blocked to unblocked code.
		nx = maxint(0, Ilaenv(func() *int { y := 3; return &y }(), []byte("ZGEQRF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1)))
		if nx < k {
			//           Determine if workspace is large enough for blocked code.
			ldwork = (*n)
			iws = ldwork * nb
			if (*lwork) < iws {
				//              Not enough workspace to use optimal NB:  reduce NB and
				//              determine the minimum value of NB.
				nb = (*lwork) / ldwork
				nbmin = maxint(2, Ilaenv(func() *int { y := 2; return &y }(), []byte("ZGEQRF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1)))
			}
		}
	}

	if nb >= nbmin && nb < k && nx < k {
		//        Use blocked code initially
		for i = 1; i <= k-nx; i += nb {
			ib = minint(k-i+1, nb)

			//           Compute the QR factorization of the current block
			//           A(i:m,i:i+ib-1)
			Zgeqr2(toPtr((*m)-i+1), &ib, a.Off(i-1, i-1), lda, tau.Off(i-1), work, &iinfo)
			if i+ib <= (*n) {
				//              Form the triangular factor of the block reflector
				//              H = H(i) H(i+1) . . . H(i+ib-1)
				Zlarft('F', 'C', toPtr((*m)-i+1), &ib, a.Off(i-1, i-1), lda, tau.Off(i-1), work.CMatrix(ldwork, opts), &ldwork)

				//              Apply H**H to A(i:m,i+ib:n) from the left
				Zlarfb('L', 'C', 'F', 'C', toPtr((*m)-i+1), toPtr((*n)-i-ib+1), &ib, a.Off(i-1, i-1), lda, work.CMatrix(ldwork, opts), &ldwork, a.Off(i-1, i+ib-1), lda, work.CMatrixOff(ib+1-1, ldwork, opts), &ldwork)
			}
		}
	} else {
		i = 1
	}

	//     Use unblocked code to factor the last or only block.
	if i <= k {
		Zgeqr2(toPtr((*m)-i+1), toPtr((*n)-i+1), a.Off(i-1, i-1), lda, tau.Off(i-1), work, &iinfo)
	}

	work.SetRe(0, float64(iws))
}
