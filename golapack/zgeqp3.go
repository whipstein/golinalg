package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgeqp3 computes a QR factorization with column pivoting of a
// matrix A:  A*P = Q*R  using Level 3 BLAS.
func Zgeqp3(m, n int, a *mat.CMatrix, jpvt *[]int, tau, work *mat.CVector, lwork int, rwork *mat.Vector) (err error) {
	var lquery bool
	var fjb, inb, inbmin, iws, ixover, j, jb, lwkopt, minmn, minws, na, nb, nbmin, nfxd, nx, sm, sminmn, sn, topbmn int

	inb = 1
	inbmin = 2
	ixover = 3

	//     Test input arguments
	//  ====================
	lquery = (lwork == -1)
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}

	if err == nil {
		minmn = min(m, n)
		if minmn == 0 {
			iws = 1
			lwkopt = 1
		} else {
			iws = n + 1
			nb = Ilaenv(inb, "Zgeqrf", []byte{' '}, m, n, -1, -1)
			lwkopt = (n + 1) * nb
		}
		work.SetRe(0, float64(lwkopt))

		if (lwork < iws) && !lquery {
			err = fmt.Errorf("(lwork < iws) && !lquery: lwork=%v, iws=%v, lquery=%v", lwork, iws, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zgeqp3", err)
		return
	} else if lquery {
		return
	}

	//     Move initial columns up front.
	nfxd = 1
	for j = 1; j <= n; j++ {
		if (*jpvt)[j-1] != 0 {
			if j != nfxd {
				a.Off(0, nfxd-1).CVector().Swap(m, a.Off(0, j-1).CVector(), 1, 1)
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
		na = min(m, nfxd)
		//CC      CALL ZGEQR2( M, NA, A, LDA, TAU, WORK, INFO )
		if err = Zgeqrf(m, na, a, tau, work, lwork); err != nil {
			panic(err)
		}
		iws = max(iws, int(work.GetRe(0)))
		if na < n {
			//CC         CALL ZUNM2R( 'Left', 'Conjugate Transpose', M, N-NA,
			//CC  $                   NA, A, LDA, TAU, A( 1, NA+1 ), LDA, WORK,
			//CC  $                   INFO )
			if err = Zunmqr(Left, ConjTrans, m, n-na, na, a, tau, a.Off(0, na), work, lwork); err != nil {
				panic(err)
			}
			iws = max(iws, int(work.GetRe(0)))
		}
	}

	//     Factorize free columns
	//  ======================
	if nfxd < minmn {

		sm = m - nfxd
		sn = n - nfxd
		sminmn = minmn - nfxd

		//        Determine the block size.
		nb = Ilaenv(inb, "Zgeqrf", []byte{' '}, sm, sn, -1, -1)
		nbmin = 2
		nx = 0

		if (nb > 1) && (nb < sminmn) {
			//           Determine when to cross over from blocked to unblocked code.
			nx = max(0, Ilaenv(ixover, "Zgeqrf", []byte{' '}, sm, sn, -1, -1))

			if nx < sminmn {
				//              Determine if workspace is large enough for blocked code.
				minws = (sn + 1) * nb
				iws = max(iws, minws)
				if lwork < minws {
					//                 Not enough workspace to use optimal NB: Reduce NB and
					//                 determine the minimum value of NB.
					nb = lwork / (sn + 1)
					nbmin = max(2, Ilaenv(inbmin, "Zgeqrf", []byte{' '}, sm, sn, -1, -1))

				}
			}
		}

		//        Initialize partial column norms. The first N elements of work
		//        store the exact column norms.
		for j = nfxd + 1; j <= n; j++ {
			rwork.Set(j-1, a.Off(nfxd, j-1).CVector().Nrm2(sm, 1))
			rwork.Set(n+j-1, rwork.Get(j-1))
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
				jb = min(nb, topbmn-j+1)

				//              Factorize JB columns among columns J:N.
				fjb = Zlaqps(m, n-j+1, j-1, jb, a.Off(0, j-1), toSlice(jpvt, j-1), tau.Off(j-1), rwork.Off(j-1), rwork.Off(n+j-1), work, work.Off(jb).CMatrix(n-j+1, opts))

				j = j + fjb
				goto label30
			}
		} else {
			j = nfxd + 1
		}

		//        Use unblocked code to factor the last or only block.
		if j <= minmn {
			Zlaqp2(m, n-j+1, j-1, a.Off(0, j-1), toSlice(jpvt, j-1), tau.Off(j-1), rwork.Off(j-1), rwork.Off(n+j-1), work)
		}

	}

	work.SetRe(0, float64(lwkopt))

	return
}
