package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zgeqp3 computes a QR factorization with column pivoting of a
// matrix A:  A*P = Q*R  using Level 3 BLAS.
func Zgeqp3(m, n *int, a *mat.CMatrix, lda *int, jpvt *[]int, tau, work *mat.CVector, lwork *int, rwork *mat.Vector, info *int) {
	var lquery bool
	var fjb, inb, inbmin, iws, ixover, j, jb, lwkopt, minmn, minws, na, nb, nbmin, nfxd, nx, sm, sminmn, sn, topbmn int

	inb = 1
	inbmin = 2
	ixover = 3

	//     Test input arguments
	//  ====================
	(*info) = 0
	lquery = ((*lwork) == -1)
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *m) {
		(*info) = -4
	}

	if (*info) == 0 {
		minmn = minint(*m, *n)
		if minmn == 0 {
			iws = 1
			lwkopt = 1
		} else {
			iws = (*n) + 1
			nb = Ilaenv(&inb, []byte("ZGEQRF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
			lwkopt = ((*n) + 1) * nb
		}
		work.SetRe(0, float64(lwkopt))

		if ((*lwork) < iws) && !lquery {
			(*info) = -8
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGEQP3"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Move initial columns up front.
	nfxd = 1
	for j = 1; j <= (*n); j++ {
		if (*jpvt)[j-1] != 0 {
			if j != nfxd {
				goblas.Zswap(m, a.CVector(0, j-1), func() *int { y := 1; return &y }(), a.CVector(0, nfxd-1), func() *int { y := 1; return &y }())
				(*jpvt)[j-1] = (*jpvt)[nfxd-1]
				(*jpvt)[nfxd-1] = j
			} else {
				(*jpvt)[j-1] = j
			}
			nfxd = nfxd + 1
		} else {
			(*jpvt)[j-1] = j
		}
	}
	nfxd = nfxd - 1

	//     Factorize fixed columns
	//  =======================
	//
	//     Compute the QR factorization of fixed columns and update
	//     remaining columns.
	if nfxd > 0 {
		na = minint(*m, nfxd)
		//CC      CALL ZGEQR2( M, NA, A, LDA, TAU, WORK, INFO )
		Zgeqrf(m, &na, a, lda, tau, work, lwork, info)
		iws = maxint(iws, int(work.GetRe(0)))
		if na < (*n) {
			//CC         CALL ZUNM2R( 'Left', 'Conjugate Transpose', M, N-NA,
			//CC  $                   NA, A, LDA, TAU, A( 1, NA+1 ), LDA, WORK,
			//CC  $                   INFO )
			Zunmqr('L', 'C', m, toPtr((*n)-na), &na, a, lda, tau, a.Off(0, na+1-1), lda, work, lwork, info)
			iws = maxint(iws, int(work.GetRe(0)))
		}
	}

	//     Factorize free columns
	//  ======================
	if nfxd < minmn {

		sm = (*m) - nfxd
		sn = (*n) - nfxd
		sminmn = minmn - nfxd

		//        Determine the block size.
		nb = Ilaenv(&inb, []byte("ZGEQRF"), []byte{' '}, &sm, &sn, toPtr(-1), toPtr(-1))
		nbmin = 2
		nx = 0

		if (nb > 1) && (nb < sminmn) {
			//           Determine when to cross over from blocked to unblocked code.
			nx = maxint(0, Ilaenv(&ixover, []byte("ZGEQRF"), []byte{' '}, &sm, &sn, toPtr(-1), toPtr(-1)))

			if nx < sminmn {
				//              Determine if workspace is large enough for blocked code.
				minws = (sn + 1) * nb
				iws = maxint(iws, minws)
				if (*lwork) < minws {
					//                 Not enough workspace to use optimal NB: Reduce NB and
					//                 determine the minimum value of NB.
					nb = (*lwork) / (sn + 1)
					nbmin = maxint(2, Ilaenv(&inbmin, []byte("ZGEQRF"), []byte{' '}, &sm, &sn, toPtr(-1), toPtr(-1)))

				}
			}
		}

		//        Initialize partial column norms. The first N elements of work
		//        store the exact column norms.
		for j = nfxd + 1; j <= (*n); j++ {
			rwork.Set(j-1, goblas.Dznrm2(&sm, a.CVector(nfxd+1-1, j-1), func() *int { y := 1; return &y }()))
			rwork.Set((*n)+j-1, rwork.Get(j-1))
		}

		if (nb >= nbmin) && (nb < sminmn) && (nx < sminmn) {
			//           Use blocked code initially.
			j = nfxd + 1

			//           Compute factorization: while loop.
			//
			topbmn = minmn - nx
		label30:
			;
			if j <= topbmn {
				jb = minint(nb, topbmn-j+1)

				//              Factorize JB columns among columns J:N.
				Zlaqps(m, toPtr((*n)-j+1), toPtr(j-1), &jb, &fjb, a.Off(0, j-1), lda, toSlice(jpvt, j-1), tau.Off(j-1), rwork.Off(j-1), rwork.Off((*n)+j-1), work, work.CMatrixOff(jb+1-1, (*n)-j+1, opts), toPtr((*n)-j+1))

				j = j + fjb
				goto label30
			}
		} else {
			j = nfxd + 1
		}

		//        Use unblocked code to factor the last or only block.
		if j <= minmn {
			Zlaqp2(m, toPtr((*n)-j+1), toPtr(j-1), a.Off(0, j-1), lda, toSlice(jpvt, j-1), tau.Off(j-1), rwork.Off(j-1), rwork.Off((*n)+j-1), work.Off(0))
		}

	}

	work.SetRe(0, float64(lwkopt))
}
