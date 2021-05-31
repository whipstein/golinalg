package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgesdd computes the singular value decomposition (SVD) of a complex
// M-by-N matrix A, optionally computing the left and/or right singular
// vectors, by using divide-and-conquer method. The SVD is written
//
//      A = U * SIGMA * conjugate-transpose(V)
//
// where SIGMA is an M-by-N matrix which is zero except for its
// minint(m,n) diagonal elements, U is an M-by-M unitary matrix, and
// V is an N-by-N unitary matrix.  The diagonal elements of SIGMA
// are the singular values of A; they are real and non-negative, and
// are returned in descending order.  The first minint(m,n) columns of
// U and V are the left and right singular vectors of A.
//
// Note that the routine returns VT = V**H, not V.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Zgesdd(jobz byte, m, n *int, a *mat.CMatrix, lda *int, s *mat.Vector, u *mat.CMatrix, ldu *int, vt *mat.CMatrix, ldvt *int, work *mat.CVector, lwork *int, rwork *mat.Vector, iwork *[]int, info *int) {
	var lquery, wntqa, wntqas, wntqn, wntqo, wntqs bool
	var cone, czero complex128
	var anrm, bignum, eps, one, smlnum, zero float64
	var blk, chunk, i, ie, ierr, il, ir, iru, irvt, iscl, itau, itaup, itauq, iu, ivt, ldwkvt, ldwrkl, ldwrkr, ldwrku, lworkZgebrdMm, lworkZgebrdMn, lworkZgebrdNn, lworkZgelqfMn, lworkZgeqrfMn, lworkZungbrPMn, lworkZungbrPNn, lworkZungbrQMm, lworkZungbrQMn, lworkZunglqMn, lworkZunglqNn, lworkZungqrMm, lworkZungqrMn, lworkZunmbrPrcMm, lworkZunmbrPrcMn, lworkZunmbrPrcNn, lworkZunmbrQlnMm, lworkZunmbrQlnMn, lworkZunmbrQlnNn, maxwrk, minmn, minwrk, mnthr1, mnthr2, nrwork, nwork, wrkbl int
	cdum := cvf(100)
	dum := vf(100)
	idum := make([]int, 1)

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	zero = 0.0
	one = 1.0

	//     Test the input arguments
	(*info) = 0
	minmn = minint(*m, *n)
	mnthr1 = int(float64(minmn) * 17.0 / 9.0)
	mnthr2 = int(float64(minmn) * 5.0 / 3.0)
	wntqa = jobz == 'A'
	wntqs = jobz == 'S'
	wntqas = wntqa || wntqs
	wntqo = jobz == 'O'
	wntqn = jobz == 'N'
	lquery = ((*lwork) == -1)
	minwrk = 1
	maxwrk = 1

	if !(wntqa || wntqs || wntqo || wntqn) {
		(*info) = -1
	} else if (*m) < 0 {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *m) {
		(*info) = -5
	} else if (*ldu) < 1 || (wntqas && (*ldu) < (*m)) || (wntqo && (*m) < (*n) && (*ldu) < (*m)) {
		(*info) = -8
	} else if (*ldvt) < 1 || (wntqa && (*ldvt) < (*n)) || (wntqs && (*ldvt) < minmn) || (wntqo && (*m) >= (*n) && (*ldvt) < (*n)) {
		(*info) = -10
	}

	//     Compute workspace
	//       Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace allocated at that point in the code,
	//       as well as the preferred amount for good performance.
	//       CWorkspace refers to complex workspace, and RWorkspace to
	//       real workspace. NB refers to the optimal block size for the
	//       immediately following subroutine, as returned by ILAENV.)
	if (*info) == 0 {
		minwrk = 1
		maxwrk = 1
		if (*m) >= (*n) && minmn > 0 {
			//           There is no complex work space needed for bidiagonal SVD
			//           The real work space needed for bidiagonal SVD (dbdsdc) is
			//           BDSPAC = 3*N*N + 4*N for singular values and vectors;
			//           BDSPAC = 4*N         for singular values only;
			//           not including e, RU, and RVT matrices.
			//
			//           Compute space preferred for each routine
			Zgebrd(m, n, cdum.CMatrix(*m, opts), m, dum.Off(0), dum.Off(0), cdum.Off(0), cdum.Off(0), cdum.Off(0), toPtr(-1), &ierr)
			lworkZgebrdMn = int(cdum.GetRe(0))

			Zgebrd(n, n, cdum.CMatrix(*n, opts), n, dum.Off(0), dum.Off(0), cdum.Off(0), cdum.Off(0), cdum.Off(0), toPtr(-1), &ierr)
			lworkZgebrdNn = int(cdum.GetRe(0))

			Zgeqrf(m, n, cdum.CMatrix(*m, opts), m, cdum.Off(0), cdum.Off(0), toPtr(-1), &ierr)
			lworkZgeqrfMn = int(cdum.GetRe(0))

			Zungbr('P', n, n, n, cdum.CMatrix(*n, opts), n, cdum.Off(0), cdum.Off(0), toPtr(-1), &ierr)
			lworkZungbrPNn = int(cdum.GetRe(0))

			Zungbr('Q', m, m, n, cdum.CMatrix(*m, opts), m, cdum.Off(0), cdum.Off(0), toPtr(-1), &ierr)
			lworkZungbrQMm = int(cdum.GetRe(0))

			Zungbr('Q', m, n, n, cdum.CMatrix(*m, opts), m, cdum.Off(0), cdum.Off(0), toPtr(-1), &ierr)
			lworkZungbrQMn = int(cdum.GetRe(0))

			Zungqr(m, m, n, cdum.CMatrix(*m, opts), m, cdum.Off(0), cdum.Off(0), toPtr(-1), &ierr)
			lworkZungqrMm = int(cdum.GetRe(0))

			Zungqr(m, n, n, cdum.CMatrix(*m, opts), m, cdum.Off(0), cdum.Off(0), toPtr(-1), &ierr)
			lworkZungqrMn = int(cdum.GetRe(0))

			Zunmbr('P', 'R', 'C', n, n, n, cdum.CMatrix(*n, opts), n, cdum.Off(0), cdum.CMatrix(*n, opts), n, cdum.Off(0), toPtr(-1), &ierr)
			lworkZunmbrPrcNn = int(cdum.GetRe(0))

			Zunmbr('Q', 'L', 'N', m, m, n, cdum.CMatrix(*m, opts), m, cdum.Off(0), cdum.CMatrix(*m, opts), m, cdum.Off(0), toPtr(-1), &ierr)
			lworkZunmbrQlnMm = int(cdum.GetRe(0))

			Zunmbr('Q', 'L', 'N', m, n, n, cdum.CMatrix(*m, opts), m, cdum.Off(0), cdum.CMatrix(*m, opts), m, cdum.Off(0), toPtr(-1), &ierr)
			lworkZunmbrQlnMn = int(cdum.GetRe(0))

			Zunmbr('Q', 'L', 'N', n, n, n, cdum.CMatrix(*n, opts), n, cdum.Off(0), cdum.CMatrix(*m, opts), n, cdum.Off(0), toPtr(-1), &ierr)
			lworkZunmbrQlnNn = int(cdum.GetRe(0))

			if (*m) >= mnthr1 {
				if wntqn {
					//                 Path 1 (M >> N, JOBZ='N')
					maxwrk = (*n) + lworkZgeqrfMn
					maxwrk = maxint(maxwrk, 2*(*n)+lworkZgebrdNn)
					minwrk = 3 * (*n)
				} else if wntqo {
					//                 Path 2 (M >> N, JOBZ='O')
					wrkbl = (*n) + lworkZgeqrfMn
					wrkbl = maxint(wrkbl, (*n)+lworkZungqrMn)
					wrkbl = maxint(wrkbl, 2*(*n)+lworkZgebrdNn)
					wrkbl = maxint(wrkbl, 2*(*n)+lworkZunmbrQlnNn)
					wrkbl = maxint(wrkbl, 2*(*n)+lworkZunmbrPrcNn)
					maxwrk = (*m)*(*n) + (*n)*(*n) + wrkbl
					minwrk = 2*(*n)*(*n) + 3*(*n)
				} else if wntqs {
					//                 Path 3 (M >> N, JOBZ='S')
					wrkbl = (*n) + lworkZgeqrfMn
					wrkbl = maxint(wrkbl, (*n)+lworkZungqrMn)
					wrkbl = maxint(wrkbl, 2*(*n)+lworkZgebrdNn)
					wrkbl = maxint(wrkbl, 2*(*n)+lworkZunmbrQlnNn)
					wrkbl = maxint(wrkbl, 2*(*n)+lworkZunmbrPrcNn)
					maxwrk = (*n)*(*n) + wrkbl
					minwrk = (*n)*(*n) + 3*(*n)
				} else if wntqa {
					//                 Path 4 (M >> N, JOBZ='A')
					wrkbl = (*n) + lworkZgeqrfMn
					wrkbl = maxint(wrkbl, (*n)+lworkZungqrMm)
					wrkbl = maxint(wrkbl, 2*(*n)+lworkZgebrdNn)
					wrkbl = maxint(wrkbl, 2*(*n)+lworkZunmbrQlnNn)
					wrkbl = maxint(wrkbl, 2*(*n)+lworkZunmbrPrcNn)
					maxwrk = (*n)*(*n) + wrkbl
					minwrk = (*n)*(*n) + maxint(3*(*n), (*n)+(*m))
				}
			} else if (*m) >= mnthr2 {
				//              Path 5 (M >> N, but not as much as MNTHR1)
				maxwrk = 2*(*n) + lworkZgebrdMn
				minwrk = 2*(*n) + (*m)
				if wntqo {
					//                 Path 5o (M >> N, JOBZ='O')
					maxwrk = maxint(maxwrk, 2*(*n)+lworkZungbrPNn)
					maxwrk = maxint(maxwrk, 2*(*n)+lworkZungbrQMn)
					maxwrk = maxwrk + (*m)*(*n)
					minwrk = minwrk + (*n)*(*n)
				} else if wntqs {
					//                 Path 5s (M >> N, JOBZ='S')
					maxwrk = maxint(maxwrk, 2*(*n)+lworkZungbrPNn)
					maxwrk = maxint(maxwrk, 2*(*n)+lworkZungbrQMn)
				} else if wntqa {
					//                 Path 5a (M >> N, JOBZ='A')
					maxwrk = maxint(maxwrk, 2*(*n)+lworkZungbrPNn)
					maxwrk = maxint(maxwrk, 2*(*n)+lworkZungbrQMm)
				}
			} else {
				//              Path 6 (M >= N, but not much larger)
				maxwrk = 2*(*n) + lworkZgebrdMn
				minwrk = 2*(*n) + (*m)
				if wntqo {
					//                 Path 6o (M >= N, JOBZ='O')
					maxwrk = maxint(maxwrk, 2*(*n)+lworkZunmbrPrcNn)
					maxwrk = maxint(maxwrk, 2*(*n)+lworkZunmbrQlnMn)
					maxwrk = maxwrk + (*m)*(*n)
					minwrk = minwrk + (*n)*(*n)
				} else if wntqs {
					//                 Path 6s (M >= N, JOBZ='S')
					maxwrk = maxint(maxwrk, 2*(*n)+lworkZunmbrQlnMn)
					maxwrk = maxint(maxwrk, 2*(*n)+lworkZunmbrPrcNn)
				} else if wntqa {
					//                 Path 6a (M >= N, JOBZ='A')
					maxwrk = maxint(maxwrk, 2*(*n)+lworkZunmbrQlnMm)
					maxwrk = maxint(maxwrk, 2*(*n)+lworkZunmbrPrcNn)
				}
			}
		} else if minmn > 0 {
			//           There is no complex work space needed for bidiagonal SVD
			//           The real work space needed for bidiagonal SVD (dbdsdc) is
			//           BDSPAC = 3*M*M + 4*M for singular values and vectors;
			//           BDSPAC = 4*M         for singular values only;
			//           not including e, RU, and RVT matrices.
			//
			//           Compute space preferred for each routine
			Zgebrd(m, n, cdum.CMatrix(*m, opts), m, dum.Off(0), dum.Off(0), cdum.Off(0), cdum.Off(0), cdum.Off(0), toPtr(-1), &ierr)
			lworkZgebrdMn = int(cdum.GetRe(0))

			Zgebrd(m, m, cdum.CMatrix(*m, opts), m, dum.Off(0), dum.Off(0), cdum.Off(0), cdum.Off(0), cdum.Off(0), toPtr(-1), &ierr)
			lworkZgebrdMm = int(cdum.GetRe(0))

			Zgelqf(m, n, cdum.CMatrix(*m, opts), m, cdum.Off(0), cdum.Off(0), toPtr(-1), &ierr)
			lworkZgelqfMn = int(cdum.GetRe(0))

			Zungbr('P', m, n, m, cdum.CMatrix(*m, opts), m, cdum.Off(0), cdum.Off(0), toPtr(-1), &ierr)
			lworkZungbrPMn = int(cdum.GetRe(0))

			Zungbr('P', n, n, m, cdum.CMatrix(*n, opts), n, cdum.Off(0), cdum.Off(0), toPtr(-1), &ierr)
			lworkZungbrPNn = int(cdum.GetRe(0))

			Zungbr('Q', m, m, n, cdum.CMatrix(*m, opts), m, cdum.Off(0), cdum.Off(0), toPtr(-1), &ierr)
			lworkZungbrQMm = int(cdum.GetRe(0))

			Zunglq(m, n, m, cdum.CMatrix(*m, opts), m, cdum.Off(0), cdum.Off(0), toPtr(-1), &ierr)
			lworkZunglqMn = int(cdum.GetRe(0))

			Zunglq(n, n, m, cdum.CMatrix(*n, opts), n, cdum.Off(0), cdum.Off(0), toPtr(-1), &ierr)
			lworkZunglqNn = int(cdum.GetRe(0))

			Zunmbr('P', 'R', 'C', m, m, m, cdum.CMatrix(*m, opts), m, cdum.Off(0), cdum.CMatrix(*m, opts), m, cdum.Off(0), toPtr(-1), &ierr)
			lworkZunmbrPrcMm = int(cdum.GetRe(0))

			Zunmbr('P', 'R', 'C', m, n, m, cdum.CMatrix(*m, opts), m, cdum.Off(0), cdum.CMatrix(*m, opts), m, cdum.Off(0), toPtr(-1), &ierr)
			lworkZunmbrPrcMn = int(cdum.GetRe(0))

			Zunmbr('P', 'R', 'C', n, n, m, cdum.CMatrix(*n, opts), n, cdum.Off(0), cdum.CMatrix(*n, opts), n, cdum.Off(0), toPtr(-1), &ierr)
			lworkZunmbrPrcNn = int(cdum.GetRe(0))

			Zunmbr('Q', 'L', 'N', m, m, m, cdum.CMatrix(*m, opts), m, cdum.Off(0), cdum.CMatrix(*m, opts), m, cdum.Off(0), toPtr(-1), &ierr)
			lworkZunmbrQlnMm = int(cdum.GetRe(0))

			if (*n) >= mnthr1 {
				if wntqn {
					//                 Path 1t (N >> M, JOBZ='N')
					maxwrk = (*m) + lworkZgelqfMn
					maxwrk = maxint(maxwrk, 2*(*m)+lworkZgebrdMm)
					minwrk = 3 * (*m)
				} else if wntqo {
					//                 Path 2t (N >> M, JOBZ='O')
					wrkbl = (*m) + lworkZgelqfMn
					wrkbl = maxint(wrkbl, (*m)+lworkZunglqMn)
					wrkbl = maxint(wrkbl, 2*(*m)+lworkZgebrdMm)
					wrkbl = maxint(wrkbl, 2*(*m)+lworkZunmbrQlnMm)
					wrkbl = maxint(wrkbl, 2*(*m)+lworkZunmbrPrcMm)
					maxwrk = (*m)*(*n) + (*m)*(*m) + wrkbl
					minwrk = 2*(*m)*(*m) + 3*(*m)
				} else if wntqs {
					//                 Path 3t (N >> M, JOBZ='S')
					wrkbl = (*m) + lworkZgelqfMn
					wrkbl = maxint(wrkbl, (*m)+lworkZunglqMn)
					wrkbl = maxint(wrkbl, 2*(*m)+lworkZgebrdMm)
					wrkbl = maxint(wrkbl, 2*(*m)+lworkZunmbrQlnMm)
					wrkbl = maxint(wrkbl, 2*(*m)+lworkZunmbrPrcMm)
					maxwrk = (*m)*(*m) + wrkbl
					minwrk = (*m)*(*m) + 3*(*m)
				} else if wntqa {
					//                 Path 4t (N >> M, JOBZ='A')
					wrkbl = (*m) + lworkZgelqfMn
					wrkbl = maxint(wrkbl, (*m)+lworkZunglqNn)
					wrkbl = maxint(wrkbl, 2*(*m)+lworkZgebrdMm)
					wrkbl = maxint(wrkbl, 2*(*m)+lworkZunmbrQlnMm)
					wrkbl = maxint(wrkbl, 2*(*m)+lworkZunmbrPrcMm)
					maxwrk = (*m)*(*m) + wrkbl
					minwrk = (*m)*(*m) + maxint(3*(*m), (*m)+(*n))
				}
			} else if (*n) >= mnthr2 {
				//              Path 5t (N >> M, but not as much as MNTHR1)
				maxwrk = 2*(*m) + lworkZgebrdMn
				minwrk = 2*(*m) + (*n)
				if wntqo {
					//                 Path 5to (N >> M, JOBZ='O')
					maxwrk = maxint(maxwrk, 2*(*m)+lworkZungbrQMm)
					maxwrk = maxint(maxwrk, 2*(*m)+lworkZungbrPMn)
					maxwrk = maxwrk + (*m)*(*n)
					minwrk = minwrk + (*m)*(*m)
				} else if wntqs {
					//                 Path 5ts (N >> M, JOBZ='S')
					maxwrk = maxint(maxwrk, 2*(*m)+lworkZungbrQMm)
					maxwrk = maxint(maxwrk, 2*(*m)+lworkZungbrPMn)
				} else if wntqa {
					//                 Path 5ta (N >> M, JOBZ='A')
					maxwrk = maxint(maxwrk, 2*(*m)+lworkZungbrQMm)
					maxwrk = maxint(maxwrk, 2*(*m)+lworkZungbrPNn)
				}
			} else {
				//              Path 6t (N > M, but not much larger)
				maxwrk = 2*(*m) + lworkZgebrdMn
				minwrk = 2*(*m) + (*n)
				if wntqo {
					//                 Path 6to (N > M, JOBZ='O')
					maxwrk = maxint(maxwrk, 2*(*m)+lworkZunmbrQlnMm)
					maxwrk = maxint(maxwrk, 2*(*m)+lworkZunmbrPrcMn)
					maxwrk = maxwrk + (*m)*(*n)
					minwrk = minwrk + (*m)*(*m)
				} else if wntqs {
					//                 Path 6ts (N > M, JOBZ='S')
					maxwrk = maxint(maxwrk, 2*(*m)+lworkZunmbrQlnMm)
					maxwrk = maxint(maxwrk, 2*(*m)+lworkZunmbrPrcMn)
				} else if wntqa {
					//                 Path 6ta (N > M, JOBZ='A')
					maxwrk = maxint(maxwrk, 2*(*m)+lworkZunmbrQlnMm)
					maxwrk = maxint(maxwrk, 2*(*m)+lworkZunmbrPrcNn)
				}
			}
		}
		maxwrk = maxint(maxwrk, minwrk)
	}
	if (*info) == 0 {
		work.SetRe(0, float64(maxwrk))
		if (*lwork) < minwrk && !lquery {
			(*info) = -12
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGESDD"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		return
	}

	//     Get machine constants
	eps = Dlamch(Precision)
	smlnum = math.Sqrt(Dlamch(SafeMinimum)) / eps
	bignum = one / smlnum

	//     Scale A if maxint element outside range [SMLNUM,BIGNUM]
	anrm = Zlange('M', m, n, a, lda, dum)
	iscl = 0
	if anrm > zero && anrm < smlnum {
		iscl = 1
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &smlnum, m, n, a, lda, &ierr)
	} else if anrm > bignum {
		iscl = 1
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &bignum, m, n, a, lda, &ierr)
	}

	if (*m) >= (*n) {
		//        A has at least as many rows as columns. If A has sufficiently
		//        more rows than columns, first reduce using the QR
		//        decomposition (if sufficient workspace available)
		if (*m) >= mnthr1 {

			if wntqn {
				//              Path 1 (M >> N, JOBZ='N')
				//              No singular vectors to be computed
				itau = 1
				nwork = itau + (*n)

				//              Compute A=Q*R
				//              CWorkspace: need   N [tau] + N    [work]
				//              CWorkspace: prefer N [tau] + N*NB [work]
				//              RWorkspace: need   0
				Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Zero out below R
				Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, a.Off(1, 0), lda)
				ie = 1
				itauq = 1
				itaup = itauq + (*n)
				nwork = itaup + (*n)

				//              Bidiagonalize R in A
				//              CWorkspace: need   2*N [tauq, taup] + N      [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + 2*N*NB [work]
				//              RWorkspace: need   N [e]
				Zgebrd(n, n, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				nrwork = ie + (*n)

				//              Perform bidiagonal SVD, compute singular values only
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + BDSPAC
				Dbdsdc('U', 'N', n, s, rwork.Off(ie-1), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), dum, &idum, rwork.Off(nrwork-1), iwork, info)

			} else if wntqo {
				//              Path 2 (M >> N, JOBZ='O')
				//              N left singular vectors to be overwritten on A and
				//              N right singular vectors to be computed in VT
				iu = 1

				//              WORK(IU) is N by N
				ldwrku = (*n)
				ir = iu + ldwrku*(*n)
				if (*lwork) >= (*m)*(*n)+(*n)*(*n)+3*(*n) {
					//                 WORK(IR) is M by N
					ldwrkr = (*m)
				} else {
					ldwrkr = ((*lwork) - (*n)*(*n) - 3*(*n)) / (*n)
				}
				itau = ir + ldwrkr*(*n)
				nwork = itau + (*n)

				//              Compute A=Q*R
				//              CWorkspace: need   N*N [U] + N*N [R] + N [tau] + N    [work]
				//              CWorkspace: prefer N*N [U] + N*N [R] + N [tau] + N*NB [work]
				//              RWorkspace: need   0
				Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy R to WORK( IR ), zeroing out below it
				Zlacpy('U', n, n, a, lda, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr)
				Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, work.CMatrixOff(ir+1-1, ldwrkr, opts), &ldwrkr)

				//              Generate Q in A
				//              CWorkspace: need   N*N [U] + N*N [R] + N [tau] + N    [work]
				//              CWorkspace: prefer N*N [U] + N*N [R] + N [tau] + N*NB [work]
				//              RWorkspace: need   0
				Zungqr(m, n, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				ie = 1
				itauq = itau
				itaup = itauq + (*n)
				nwork = itaup + (*n)

				//              Bidiagonalize R in WORK(IR)
				//              CWorkspace: need   N*N [U] + N*N [R] + 2*N [tauq, taup] + N      [work]
				//              CWorkspace: prefer N*N [U] + N*N [R] + 2*N [tauq, taup] + 2*N*NB [work]
				//              RWorkspace: need   N [e]
				Zgebrd(n, n, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of R in WORK(IRU) and computing right singular vectors
				//              of R in WORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
				iru = ie + (*n)
				irvt = iru + (*n)*(*n)
				nrwork = irvt + (*n)*(*n)
				Dbdsdc('U', 'I', n, s, rwork.Off(ie-1), rwork.MatrixOff(iru-1, *n, opts), n, rwork.MatrixOff(irvt-1, *n, opts), n, dum, &idum, rwork.Off(nrwork-1), iwork, info)

				//              Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
				//              Overwrite WORK(IU) by the left singular vectors of R
				//              CWorkspace: need   N*N [U] + N*N [R] + 2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer N*N [U] + N*N [R] + 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacp2('F', n, n, rwork.MatrixOff(iru-1, *n, opts), n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku)
				Zunmbr('Q', 'L', 'N', n, n, n, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itauq-1), work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy real matrix RWORK(IRVT) to complex matrix VT
				//              Overwrite VT by the right singular vectors of R
				//              CWorkspace: need   N*N [U] + N*N [R] + 2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer N*N [U] + N*N [R] + 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacp2('F', n, n, rwork.MatrixOff(irvt-1, *n, opts), n, vt, ldvt)
				Zunmbr('P', 'R', 'C', n, n, n, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itaup-1), vt, ldvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Multiply Q in A by left singular vectors of R in
				//              WORK(IU), storing result in WORK(IR) and copying to A
				//              CWorkspace: need   N*N [U] + N*N [R]
				//              CWorkspace: prefer N*N [U] + M*N [R]
				//              RWorkspace: need   0
				for i = 1; i <= (*m); i += ldwrkr {
					chunk = minint((*m)-i+1, ldwrkr)
					goblas.Zgemm(NoTrans, NoTrans, &chunk, n, n, &cone, a.Off(i-1, 0), lda, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, &czero, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr)
					Zlacpy('F', &chunk, n, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, a.Off(i-1, 0), lda)
					//Label10:
				}

			} else if wntqs {
				//              Path 3 (M >> N, JOBZ='S')
				//              N left singular vectors to be computed in U and
				//              N right singular vectors to be computed in VT
				ir = 1

				//              WORK(IR) is N by N
				ldwrkr = (*n)
				itau = ir + ldwrkr*(*n)
				nwork = itau + (*n)

				//              Compute A=Q*R
				//              CWorkspace: need   N*N [R] + N [tau] + N    [work]
				//              CWorkspace: prefer N*N [R] + N [tau] + N*NB [work]
				//              RWorkspace: need   0
				Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy R to WORK(IR), zeroing out below it
				Zlacpy('U', n, n, a, lda, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr)
				Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, work.CMatrixOff(ir+1-1, ldwrkr, opts), &ldwrkr)

				//              Generate Q in A
				//              CWorkspace: need   N*N [R] + N [tau] + N    [work]
				//              CWorkspace: prefer N*N [R] + N [tau] + N*NB [work]
				//              RWorkspace: need   0
				Zungqr(m, n, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				ie = 1
				itauq = itau
				itaup = itauq + (*n)
				nwork = itaup + (*n)

				//              Bidiagonalize R in WORK(IR)
				//              CWorkspace: need   N*N [R] + 2*N [tauq, taup] + N      [work]
				//              CWorkspace: prefer N*N [R] + 2*N [tauq, taup] + 2*N*NB [work]
				//              RWorkspace: need   N [e]
				Zgebrd(n, n, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
				iru = ie + (*n)
				irvt = iru + (*n)*(*n)
				nrwork = irvt + (*n)*(*n)
				Dbdsdc('U', 'I', n, s, rwork.Off(ie-1), rwork.MatrixOff(iru-1, *n, opts), n, rwork.MatrixOff(irvt-1, *n, opts), n, dum, &idum, rwork.Off(nrwork-1), iwork, info)

				//              Copy real matrix RWORK(IRU) to complex matrix U
				//              Overwrite U by left singular vectors of R
				//              CWorkspace: need   N*N [R] + 2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer N*N [R] + 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacp2('F', n, n, rwork.MatrixOff(iru-1, *n, opts), n, u, ldu)
				Zunmbr('Q', 'L', 'N', n, n, n, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itauq-1), u, ldu, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy real matrix RWORK(IRVT) to complex matrix VT
				//              Overwrite VT by right singular vectors of R
				//              CWorkspace: need   N*N [R] + 2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer N*N [R] + 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacp2('F', n, n, rwork.MatrixOff(irvt-1, *n, opts), n, vt, ldvt)
				Zunmbr('P', 'R', 'C', n, n, n, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itaup-1), vt, ldvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Multiply Q in A by left singular vectors of R in
				//              WORK(IR), storing result in U
				//              CWorkspace: need   N*N [R]
				//              RWorkspace: need   0
				Zlacpy('F', n, n, u, ldu, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr)
				goblas.Zgemm(NoTrans, NoTrans, m, n, n, &cone, a, lda, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, &czero, u, ldu)

			} else if wntqa {
				//              Path 4 (M >> N, JOBZ='A')
				//              M left singular vectors to be computed in U and
				//              N right singular vectors to be computed in VT
				iu = 1

				//              WORK(IU) is N by N
				ldwrku = (*n)
				itau = iu + ldwrku*(*n)
				nwork = itau + (*n)

				//              Compute A=Q*R, copying result to U
				//              CWorkspace: need   N*N [U] + N [tau] + N    [work]
				//              CWorkspace: prefer N*N [U] + N [tau] + N*NB [work]
				//              RWorkspace: need   0
				Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				Zlacpy('L', m, n, a, lda, u, ldu)

				//              Generate Q in U
				//              CWorkspace: need   N*N [U] + N [tau] + M    [work]
				//              CWorkspace: prefer N*N [U] + N [tau] + M*NB [work]
				//              RWorkspace: need   0
				Zungqr(m, m, n, u, ldu, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Produce R in A, zeroing out below it
				Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, a.Off(1, 0), lda)
				ie = 1
				itauq = itau
				itaup = itauq + (*n)
				nwork = itaup + (*n)

				//              Bidiagonalize R in A
				//              CWorkspace: need   N*N [U] + 2*N [tauq, taup] + N      [work]
				//              CWorkspace: prefer N*N [U] + 2*N [tauq, taup] + 2*N*NB [work]
				//              RWorkspace: need   N [e]
				Zgebrd(n, n, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				iru = ie + (*n)
				irvt = iru + (*n)*(*n)
				nrwork = irvt + (*n)*(*n)

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
				Dbdsdc('U', 'I', n, s, rwork.Off(ie-1), rwork.MatrixOff(iru-1, *n, opts), n, rwork.MatrixOff(irvt-1, *n, opts), n, dum, &idum, rwork.Off(nrwork-1), iwork, info)

				//              Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
				//              Overwrite WORK(IU) by left singular vectors of R
				//              CWorkspace: need   N*N [U] + 2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer N*N [U] + 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacp2('F', n, n, rwork.MatrixOff(iru-1, *n, opts), n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku)
				Zunmbr('Q', 'L', 'N', n, n, n, a, lda, work.Off(itauq-1), work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy real matrix RWORK(IRVT) to complex matrix VT
				//              Overwrite VT by right singular vectors of R
				//              CWorkspace: need   N*N [U] + 2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer N*N [U] + 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacp2('F', n, n, rwork.MatrixOff(irvt-1, *n, opts), n, vt, ldvt)
				Zunmbr('P', 'R', 'C', n, n, n, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Multiply Q in U by left singular vectors of R in
				//              WORK(IU), storing result in A
				//              CWorkspace: need   N*N [U]
				//              RWorkspace: need   0
				goblas.Zgemm(NoTrans, NoTrans, m, n, n, &cone, u, ldu, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, &czero, a, lda)

				//              Copy left singular vectors of A from A to U
				Zlacpy('F', m, n, a, lda, u, ldu)

			}

		} else if (*m) >= mnthr2 {
			//           MNTHR2 <= M < MNTHR1
			//
			//           Path 5 (M >> N, but not as much as MNTHR1)
			//           Reduce to bidiagonal form without QR decomposition, use
			//           ZUNGBR and matrix multiplication to compute singular vectors
			ie = 1
			nrwork = ie + (*n)
			itauq = 1
			itaup = itauq + (*n)
			nwork = itaup + (*n)

			//           Bidiagonalize A
			//           CWorkspace: need   2*N [tauq, taup] + M        [work]
			//           CWorkspace: prefer 2*N [tauq, taup] + (M+N)*NB [work]
			//           RWorkspace: need   N [e]
			Zgebrd(m, n, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
			if wntqn {
				//              Path 5n (M >> N, JOBZ='N')
				//              Compute singular values only
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + BDSPAC
				Dbdsdc('U', 'N', n, s, rwork.Off(ie-1), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), dum, &idum, rwork.Off(nrwork-1), iwork, info)
			} else if wntqo {
				iu = nwork
				iru = nrwork
				irvt = iru + (*n)*(*n)
				nrwork = irvt + (*n)*(*n)

				//              Path 5o (M >> N, JOBZ='O')
				//              Copy A to VT, generate P**H
				//              CWorkspace: need   2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacpy('U', n, n, a, lda, vt, ldvt)
				Zungbr('P', n, n, n, vt, ldvt, work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Generate Q in A
				//              CWorkspace: need   2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zungbr('Q', m, n, n, a, lda, work.Off(itauq-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				if (*lwork) >= (*m)*(*n)+3*(*n) {
					//                 WORK( IU ) is M by N
					ldwrku = (*m)
				} else {
					//                 WORK(IU) is LDWRKU by N
					ldwrku = ((*lwork) - 3*(*n)) / (*n)
				}
				nwork = iu + ldwrku*(*n)

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
				Dbdsdc('U', 'I', n, s, rwork.Off(ie-1), rwork.MatrixOff(iru-1, *n, opts), n, rwork.MatrixOff(irvt-1, *n, opts), n, dum, &idum, rwork.Off(nrwork-1), iwork, info)

				//              Multiply real matrix RWORK(IRVT) by P**H in VT,
				//              storing the result in WORK(IU), copying to VT
				//              CWorkspace: need   2*N [tauq, taup] + N*N [U]
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + 2*N*N [rwork]
				Zlarcm(n, n, rwork.MatrixOff(irvt-1, *n, opts), n, vt, ldvt, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, rwork.Off(nrwork-1))
				Zlacpy('F', n, n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, vt, ldvt)

				//              Multiply Q in A by real matrix RWORK(IRU), storing the
				//              result in WORK(IU), copying to A
				//              CWorkspace: need   2*N [tauq, taup] + N*N [U]
				//              CWorkspace: prefer 2*N [tauq, taup] + M*N [U]
				//              RWorkspace: need   N [e] + N*N [RU] + 2*N*N [rwork]
				//              RWorkspace: prefer N [e] + N*N [RU] + 2*M*N [rwork] < N + 5*N*N since M < 2*N here
				nrwork = irvt
				for i = 1; i <= (*m); i += ldwrku {
					chunk = minint((*m)-i+1, ldwrku)
					Zlacrm(&chunk, n, a.Off(i-1, 0), lda, rwork.MatrixOff(iru-1, *n, opts), n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, rwork.Off(nrwork-1))
					Zlacpy('F', &chunk, n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, a.Off(i-1, 0), lda)
				}

			} else if wntqs {
				//              Path 5s (M >> N, JOBZ='S')
				//              Copy A to VT, generate P**H
				//              CWorkspace: need   2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacpy('U', n, n, a, lda, vt, ldvt)
				Zungbr('P', n, n, n, vt, ldvt, work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy A to U, generate Q
				//              CWorkspace: need   2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacpy('L', m, n, a, lda, u, ldu)
				Zungbr('Q', m, n, n, u, ldu, work.Off(itauq-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
				iru = nrwork
				irvt = iru + (*n)*(*n)
				nrwork = irvt + (*n)*(*n)
				Dbdsdc('U', 'I', n, s, rwork.Off(ie-1), rwork.MatrixOff(iru-1, *n, opts), n, rwork.MatrixOff(irvt-1, *n, opts), n, dum, &idum, rwork.Off(nrwork-1), iwork, info)

				//              Multiply real matrix RWORK(IRVT) by P**H in VT,
				//              storing the result in A, copying to VT
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + 2*N*N [rwork]
				Zlarcm(n, n, rwork.MatrixOff(irvt-1, *n, opts), n, vt, ldvt, a, lda, rwork.Off(nrwork-1))
				Zlacpy('F', n, n, a, lda, vt, ldvt)

				//              Multiply Q in U by real matrix RWORK(IRU), storing the
				//              result in A, copying to U
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + 2*M*N [rwork] < N + 5*N*N since M < 2*N here
				nrwork = irvt
				Zlacrm(m, n, u, ldu, rwork.MatrixOff(iru-1, *n, opts), n, a, lda, rwork.Off(nrwork-1))
				Zlacpy('F', m, n, a, lda, u, ldu)
			} else {
				//              Path 5a (M >> N, JOBZ='A')
				//              Copy A to VT, generate P**H
				//              CWorkspace: need   2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacpy('U', n, n, a, lda, vt, ldvt)
				Zungbr('P', n, n, n, vt, ldvt, work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy A to U, generate Q
				//              CWorkspace: need   2*N [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacpy('L', m, n, a, lda, u, ldu)
				Zungbr('Q', m, m, n, u, ldu, work.Off(itauq-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
				iru = nrwork
				irvt = iru + (*n)*(*n)
				nrwork = irvt + (*n)*(*n)
				Dbdsdc('U', 'I', n, s, rwork.Off(ie-1), rwork.MatrixOff(iru-1, *n, opts), n, rwork.MatrixOff(irvt-1, *n, opts), n, dum, &idum, rwork.Off(nrwork-1), iwork, info)

				//              Multiply real matrix RWORK(IRVT) by P**H in VT,
				//              storing the result in A, copying to VT
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + 2*N*N [rwork]
				Zlarcm(n, n, rwork.MatrixOff(irvt-1, *n, opts), n, vt, ldvt, a, lda, rwork.Off(nrwork-1))
				Zlacpy('F', n, n, a, lda, vt, ldvt)

				//              Multiply Q in U by real matrix RWORK(IRU), storing the
				//              result in A, copying to U
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + 2*M*N [rwork] < N + 5*N*N since M < 2*N here
				nrwork = irvt
				Zlacrm(m, n, u, ldu, rwork.MatrixOff(iru-1, *n, opts), n, a, lda, rwork.Off(nrwork-1))
				Zlacpy('F', m, n, a, lda, u, ldu)
			}

		} else {
			//           M .LT. MNTHR2
			//
			//           Path 6 (M >= N, but not much larger)
			//           Reduce to bidiagonal form without QR decomposition
			//           Use ZUNMBR to compute singular vectors
			ie = 1
			nrwork = ie + (*n)
			itauq = 1
			itaup = itauq + (*n)
			nwork = itaup + (*n)

			//           Bidiagonalize A
			//           CWorkspace: need   2*N [tauq, taup] + M        [work]
			//           CWorkspace: prefer 2*N [tauq, taup] + (M+N)*NB [work]
			//           RWorkspace: need   N [e]
			Zgebrd(m, n, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
			if wntqn {
				//              Path 6n (M >= N, JOBZ='N')
				//              Compute singular values only
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + BDSPAC
				Dbdsdc('U', 'N', n, s, rwork.Off(ie-1), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), dum, &idum, rwork.Off(nrwork-1), iwork, info)
			} else if wntqo {
				iu = nwork
				iru = nrwork
				irvt = iru + (*n)*(*n)
				nrwork = irvt + (*n)*(*n)
				if (*lwork) >= (*m)*(*n)+3*(*n) {
					//                 WORK( IU ) is M by N
					ldwrku = (*m)
				} else {
					//                 WORK( IU ) is LDWRKU by N
					ldwrku = ((*lwork) - 3*(*n)) / (*n)
				}
				nwork = iu + ldwrku*(*n)

				//              Path 6o (M >= N, JOBZ='O')
				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
				Dbdsdc('U', 'I', n, s, rwork.Off(ie-1), rwork.MatrixOff(iru-1, *n, opts), n, rwork.MatrixOff(irvt-1, *n, opts), n, dum, &idum, rwork.Off(nrwork-1), iwork, info)

				//              Copy real matrix RWORK(IRVT) to complex matrix VT
				//              Overwrite VT by right singular vectors of A
				//              CWorkspace: need   2*N [tauq, taup] + N*N [U] + N    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + N*N [U] + N*NB [work]
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
				Zlacp2('F', n, n, rwork.MatrixOff(irvt-1, *n, opts), n, vt, ldvt)
				Zunmbr('P', 'R', 'C', n, n, n, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				if (*lwork) >= (*m)*(*n)+3*(*n) {
					//                 Path 6o-fast
					//                 Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
					//                 Overwrite WORK(IU) by left singular vectors of A, copying
					//                 to A
					//                 CWorkspace: need   2*N [tauq, taup] + M*N [U] + N    [work]
					//                 CWorkspace: prefer 2*N [tauq, taup] + M*N [U] + N*NB [work]
					//                 RWorkspace: need   N [e] + N*N [RU]
					Zlaset('F', m, n, &czero, &czero, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku)
					Zlacp2('F', n, n, rwork.MatrixOff(iru-1, *n, opts), n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku)
					Zunmbr('Q', 'L', 'N', m, n, n, a, lda, work.Off(itauq-1), work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
					Zlacpy('F', m, n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, a, lda)
				} else {
					//                 Path 6o-slow
					//                 Generate Q in A
					//                 CWorkspace: need   2*N [tauq, taup] + N*N [U] + N    [work]
					//                 CWorkspace: prefer 2*N [tauq, taup] + N*N [U] + N*NB [work]
					//                 RWorkspace: need   0
					Zungbr('Q', m, n, n, a, lda, work.Off(itauq-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

					//                 Multiply Q in A by real matrix RWORK(IRU), storing the
					//                 result in WORK(IU), copying to A
					//                 CWorkspace: need   2*N [tauq, taup] + N*N [U]
					//                 CWorkspace: prefer 2*N [tauq, taup] + M*N [U]
					//                 RWorkspace: need   N [e] + N*N [RU] + 2*N*N [rwork]
					//                 RWorkspace: prefer N [e] + N*N [RU] + 2*M*N [rwork] < N + 5*N*N since M < 2*N here
					nrwork = irvt
					for i = 1; i <= (*m); i += ldwrku {
						chunk = minint((*m)-i+1, ldwrku)
						Zlacrm(&chunk, n, a.Off(i-1, 0), lda, rwork.MatrixOff(iru-1, *n, opts), n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, rwork.Off(nrwork-1))
						Zlacpy('F', &chunk, n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, a.Off(i-1, 0), lda)
					}
				}

			} else if wntqs {
				//              Path 6s (M >= N, JOBZ='S')
				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
				iru = nrwork
				irvt = iru + (*n)*(*n)
				nrwork = irvt + (*n)*(*n)
				Dbdsdc('U', 'I', n, s, rwork.Off(ie-1), rwork.MatrixOff(iru-1, *n, opts), n, rwork.MatrixOff(irvt-1, *n, opts), n, dum, &idum, rwork.Off(nrwork-1), iwork, info)

				//              Copy real matrix RWORK(IRU) to complex matrix U
				//              Overwrite U by left singular vectors of A
				//              CWorkspace: need   2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
				Zlaset('F', m, n, &czero, &czero, u, ldu)
				Zlacp2('F', n, n, rwork.MatrixOff(iru-1, *n, opts), n, u, ldu)
				Zunmbr('Q', 'L', 'N', m, n, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy real matrix RWORK(IRVT) to complex matrix VT
				//              Overwrite VT by right singular vectors of A
				//              CWorkspace: need   2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
				Zlacp2('F', n, n, rwork.MatrixOff(irvt-1, *n, opts), n, vt, ldvt)
				Zunmbr('P', 'R', 'C', n, n, n, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
			} else {
				//              Path 6a (M >= N, JOBZ='A')
				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
				iru = nrwork
				irvt = iru + (*n)*(*n)
				nrwork = irvt + (*n)*(*n)
				Dbdsdc('U', 'I', n, s, rwork.Off(ie-1), rwork.MatrixOff(iru-1, *n, opts), n, rwork.MatrixOff(irvt-1, *n, opts), n, dum, &idum, rwork.Off(nrwork-1), iwork, info)

				//              Set the right corner of U to identity matrix
				Zlaset('F', m, m, &czero, &czero, u, ldu)
				if (*m) > (*n) {
					Zlaset('F', toPtr((*m)-(*n)), toPtr((*m)-(*n)), &czero, &cone, u.Off((*n)+1-1, (*n)+1-1), ldu)
				}

				//              Copy real matrix RWORK(IRU) to complex matrix U
				//              Overwrite U by left singular vectors of A
				//              CWorkspace: need   2*N [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + M*NB [work]
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
				Zlacp2('F', n, n, rwork.MatrixOff(iru-1, *n, opts), n, u, ldu)
				Zunmbr('Q', 'L', 'N', m, m, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy real matrix RWORK(IRVT) to complex matrix VT
				//              Overwrite VT by right singular vectors of A
				//              CWorkspace: need   2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
				Zlacp2('F', n, n, rwork.MatrixOff(irvt-1, *n, opts), n, vt, ldvt)
				Zunmbr('P', 'R', 'C', n, n, n, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
			}

		}

	} else {
		//        A has more columns than rows. If A has sufficiently more
		//        columns than rows, first reduce using the LQ decomposition (if
		//        sufficient workspace available)
		if (*n) >= mnthr1 {

			if wntqn {
				//              Path 1t (N >> M, JOBZ='N')
				//              No singular vectors to be computed
				itau = 1
				nwork = itau + (*m)

				//              Compute A=L*Q
				//              CWorkspace: need   M [tau] + M    [work]
				//              CWorkspace: prefer M [tau] + M*NB [work]
				//              RWorkspace: need   0
				Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Zero out above L
				Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, a.Off(0, 1), lda)
				ie = 1
				itauq = 1
				itaup = itauq + (*m)
				nwork = itaup + (*m)

				//              Bidiagonalize L in A
				//              CWorkspace: need   2*M [tauq, taup] + M      [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + 2*M*NB [work]
				//              RWorkspace: need   M [e]
				Zgebrd(m, m, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				nrwork = ie + (*m)

				//              Perform bidiagonal SVD, compute singular values only
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + BDSPAC
				Dbdsdc('U', 'N', m, s, rwork.Off(ie-1), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), dum, &idum, rwork.Off(nrwork-1), iwork, info)

			} else if wntqo {
				//              Path 2t (N >> M, JOBZ='O')
				//              M right singular vectors to be overwritten on A and
				//              M left singular vectors to be computed in U
				ivt = 1
				ldwkvt = (*m)

				//              WORK(IVT) is M by M
				il = ivt + ldwkvt*(*m)
				if (*lwork) >= (*m)*(*n)+(*m)*(*m)+3*(*m) {
					//                 WORK(IL) M by N
					ldwrkl = (*m)
					chunk = (*n)
				} else {
					//                 WORK(IL) is M by CHUNK
					ldwrkl = (*m)
					chunk = ((*lwork) - (*m)*(*m) - 3*(*m)) / (*m)
				}
				itau = il + ldwrkl*chunk
				nwork = itau + (*m)

				//              Compute A=L*Q
				//              CWorkspace: need   M*M [VT] + M*M [L] + M [tau] + M    [work]
				//              CWorkspace: prefer M*M [VT] + M*M [L] + M [tau] + M*NB [work]
				//              RWorkspace: need   0
				Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy L to WORK(IL), zeroing about above it
				Zlacpy('L', m, m, a, lda, work.CMatrixOff(il-1, ldwrkl, opts), &ldwrkl)
				Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, work.CMatrixOff(il+ldwrkl-1, ldwrkl, opts), &ldwrkl)

				//              Generate Q in A
				//              CWorkspace: need   M*M [VT] + M*M [L] + M [tau] + M    [work]
				//              CWorkspace: prefer M*M [VT] + M*M [L] + M [tau] + M*NB [work]
				//              RWorkspace: need   0
				Zunglq(m, n, m, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				ie = 1
				itauq = itau
				itaup = itauq + (*m)
				nwork = itaup + (*m)

				//              Bidiagonalize L in WORK(IL)
				//              CWorkspace: need   M*M [VT] + M*M [L] + 2*M [tauq, taup] + M      [work]
				//              CWorkspace: prefer M*M [VT] + M*M [L] + 2*M [tauq, taup] + 2*M*NB [work]
				//              RWorkspace: need   M [e]
				Zgebrd(m, m, work.CMatrixOff(il-1, ldwrkl, opts), &ldwrkl, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RU] + M*M [RVT] + BDSPAC
				iru = ie + (*m)
				irvt = iru + (*m)*(*m)
				nrwork = irvt + (*m)*(*m)
				Dbdsdc('U', 'I', m, s, rwork.Off(ie-1), rwork.MatrixOff(iru-1, *m, opts), m, rwork.MatrixOff(irvt-1, *m, opts), m, dum, &idum, rwork.Off(nrwork-1), iwork, info)

				//              Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
				//              Overwrite WORK(IU) by the left singular vectors of L
				//              CWorkspace: need   M*M [VT] + M*M [L] + 2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer M*M [VT] + M*M [L] + 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacp2('F', m, m, rwork.MatrixOff(iru-1, *m, opts), m, u, ldu)
				Zunmbr('Q', 'L', 'N', m, m, m, work.CMatrixOff(il-1, ldwrkl, opts), &ldwrkl, work.Off(itauq-1), u, ldu, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT)
				//              Overwrite WORK(IVT) by the right singular vectors of L
				//              CWorkspace: need   M*M [VT] + M*M [L] + 2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer M*M [VT] + M*M [L] + 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacp2('F', m, m, rwork.MatrixOff(irvt-1, *m, opts), m, work.CMatrixOff(ivt-1, ldwkvt, opts), &ldwkvt)
				Zunmbr('P', 'R', 'C', m, m, m, work.CMatrixOff(il-1, ldwrkl, opts), &ldwrkl, work.Off(itaup-1), work.CMatrixOff(ivt-1, ldwkvt, opts), &ldwkvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Multiply right singular vectors of L in WORK(IL) by Q
				//              in A, storing result in WORK(IL) and copying to A
				//              CWorkspace: need   M*M [VT] + M*M [L]
				//              CWorkspace: prefer M*M [VT] + M*N [L]
				//              RWorkspace: need   0
				for i = 1; i <= (*n); i += chunk {
					blk = minint((*n)-i+1, chunk)
					goblas.Zgemm(NoTrans, NoTrans, m, &blk, m, &cone, work.CMatrixOff(ivt-1, *m, opts), m, a.Off(0, i-1), lda, &czero, work.CMatrixOff(il-1, ldwrkl, opts), &ldwrkl)
					Zlacpy('F', m, &blk, work.CMatrixOff(il-1, ldwrkl, opts), &ldwrkl, a.Off(0, i-1), lda)
				}

			} else if wntqs {
				//              Path 3t (N >> M, JOBZ='S')
				//              M right singular vectors to be computed in VT and
				//              M left singular vectors to be computed in U
				il = 1

				//              WORK(IL) is M by M
				ldwrkl = (*m)
				itau = il + ldwrkl*(*m)
				nwork = itau + (*m)

				//              Compute A=L*Q
				//              CWorkspace: need   M*M [L] + M [tau] + M    [work]
				//              CWorkspace: prefer M*M [L] + M [tau] + M*NB [work]
				//              RWorkspace: need   0
				Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy L to WORK(IL), zeroing out above it
				Zlacpy('L', m, m, a, lda, work.CMatrixOff(il-1, ldwrkl, opts), &ldwrkl)
				Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, work.CMatrixOff(il+ldwrkl-1, ldwrkl, opts), &ldwrkl)

				//              Generate Q in A
				//              CWorkspace: need   M*M [L] + M [tau] + M    [work]
				//              CWorkspace: prefer M*M [L] + M [tau] + M*NB [work]
				//              RWorkspace: need   0
				Zunglq(m, n, m, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				ie = 1
				itauq = itau
				itaup = itauq + (*m)
				nwork = itaup + (*m)

				//              Bidiagonalize L in WORK(IL)
				//              CWorkspace: need   M*M [L] + 2*M [tauq, taup] + M      [work]
				//              CWorkspace: prefer M*M [L] + 2*M [tauq, taup] + 2*M*NB [work]
				//              RWorkspace: need   M [e]
				Zgebrd(m, m, work.CMatrixOff(il-1, ldwrkl, opts), &ldwrkl, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RU] + M*M [RVT] + BDSPAC
				iru = ie + (*m)
				irvt = iru + (*m)*(*m)
				nrwork = irvt + (*m)*(*m)
				Dbdsdc('U', 'I', m, s, rwork.Off(ie-1), rwork.MatrixOff(iru-1, *m, opts), m, rwork.MatrixOff(irvt-1, *m, opts), m, dum, &idum, rwork.Off(nrwork-1), iwork, info)

				//              Copy real matrix RWORK(IRU) to complex matrix U
				//              Overwrite U by left singular vectors of L
				//              CWorkspace: need   M*M [L] + 2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer M*M [L] + 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacp2('F', m, m, rwork.MatrixOff(iru-1, *m, opts), m, u, ldu)
				Zunmbr('Q', 'L', 'N', m, m, m, work.CMatrixOff(il-1, ldwrkl, opts), &ldwrkl, work.Off(itauq-1), u, ldu, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy real matrix RWORK(IRVT) to complex matrix VT
				//              Overwrite VT by left singular vectors of L
				//              CWorkspace: need   M*M [L] + 2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer M*M [L] + 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacp2('F', m, m, rwork.MatrixOff(irvt-1, *m, opts), m, vt, ldvt)
				Zunmbr('P', 'R', 'C', m, m, m, work.CMatrixOff(il-1, ldwrkl, opts), &ldwrkl, work.Off(itaup-1), vt, ldvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy VT to WORK(IL), multiply right singular vectors of L
				//              in WORK(IL) by Q in A, storing result in VT
				//              CWorkspace: need   M*M [L]
				//              RWorkspace: need   0
				Zlacpy('F', m, m, vt, ldvt, work.CMatrixOff(il-1, ldwrkl, opts), &ldwrkl)
				goblas.Zgemm(NoTrans, NoTrans, m, n, m, &cone, work.CMatrixOff(il-1, ldwrkl, opts), &ldwrkl, a, lda, &czero, vt, ldvt)

			} else if wntqa {
				//              Path 4t (N >> M, JOBZ='A')
				//              N right singular vectors to be computed in VT and
				//              M left singular vectors to be computed in U
				ivt = 1

				//              WORK(IVT) is M by M
				ldwkvt = (*m)
				itau = ivt + ldwkvt*(*m)
				nwork = itau + (*m)

				//              Compute A=L*Q, copying result to VT
				//              CWorkspace: need   M*M [VT] + M [tau] + M    [work]
				//              CWorkspace: prefer M*M [VT] + M [tau] + M*NB [work]
				//              RWorkspace: need   0
				Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				Zlacpy('U', m, n, a, lda, vt, ldvt)

				//              Generate Q in VT
				//              CWorkspace: need   M*M [VT] + M [tau] + N    [work]
				//              CWorkspace: prefer M*M [VT] + M [tau] + N*NB [work]
				//              RWorkspace: need   0
				Zunglq(n, n, m, vt, ldvt, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Produce L in A, zeroing out above it
				Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, a.Off(0, 1), lda)
				ie = 1
				itauq = itau
				itaup = itauq + (*m)
				nwork = itaup + (*m)

				//              Bidiagonalize L in A
				//              CWorkspace: need   M*M [VT] + 2*M [tauq, taup] + M      [work]
				//              CWorkspace: prefer M*M [VT] + 2*M [tauq, taup] + 2*M*NB [work]
				//              RWorkspace: need   M [e]
				Zgebrd(m, m, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RU] + M*M [RVT] + BDSPAC
				iru = ie + (*m)
				irvt = iru + (*m)*(*m)
				nrwork = irvt + (*m)*(*m)
				Dbdsdc('U', 'I', m, s, rwork.Off(ie-1), rwork.MatrixOff(iru-1, *m, opts), m, rwork.MatrixOff(irvt-1, *m, opts), m, dum, &idum, rwork.Off(nrwork-1), iwork, info)

				//              Copy real matrix RWORK(IRU) to complex matrix U
				//              Overwrite U by left singular vectors of L
				//              CWorkspace: need   M*M [VT] + 2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer M*M [VT] + 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacp2('F', m, m, rwork.MatrixOff(iru-1, *m, opts), m, u, ldu)
				Zunmbr('Q', 'L', 'N', m, m, m, a, lda, work.Off(itauq-1), u, ldu, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT)
				//              Overwrite WORK(IVT) by right singular vectors of L
				//              CWorkspace: need   M*M [VT] + 2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer M*M [VT] + 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacp2('F', m, m, rwork.MatrixOff(irvt-1, *m, opts), m, work.CMatrixOff(ivt-1, ldwkvt, opts), &ldwkvt)
				Zunmbr('P', 'R', 'C', m, m, m, a, lda, work.Off(itaup-1), work.CMatrixOff(ivt-1, ldwkvt, opts), &ldwkvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Multiply right singular vectors of L in WORK(IVT) by
				//              Q in VT, storing result in A
				//              CWorkspace: need   M*M [VT]
				//              RWorkspace: need   0
				goblas.Zgemm(NoTrans, NoTrans, m, n, m, &cone, work.CMatrixOff(ivt-1, ldwkvt, opts), &ldwkvt, vt, ldvt, &czero, a, lda)

				//              Copy right singular vectors of A from A to VT
				Zlacpy('F', m, n, a, lda, vt, ldvt)

			}

		} else if (*n) >= mnthr2 {
			//           MNTHR2 <= N < MNTHR1
			//
			//           Path 5t (N >> M, but not as much as MNTHR1)
			//           Reduce to bidiagonal form without QR decomposition, use
			//           ZUNGBR and matrix multiplication to compute singular vectors
			ie = 1
			nrwork = ie + (*m)
			itauq = 1
			itaup = itauq + (*m)
			nwork = itaup + (*m)

			//           Bidiagonalize A
			//           CWorkspace: need   2*M [tauq, taup] + N        [work]
			//           CWorkspace: prefer 2*M [tauq, taup] + (M+N)*NB [work]
			//           RWorkspace: need   M [e]
			Zgebrd(m, n, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

			if wntqn {
				//              Path 5tn (N >> M, JOBZ='N')
				//              Compute singular values only
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + BDSPAC
				Dbdsdc('L', 'N', m, s, rwork.Off(ie-1), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), dum, &idum, rwork.Off(nrwork-1), iwork, info)
			} else if wntqo {
				irvt = nrwork
				iru = irvt + (*m)*(*m)
				nrwork = iru + (*m)*(*m)
				ivt = nwork

				//              Path 5to (N >> M, JOBZ='O')
				//              Copy A to U, generate Q
				//              CWorkspace: need   2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacpy('L', m, m, a, lda, u, ldu)
				Zungbr('Q', m, m, n, u, ldu, work.Off(itauq-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Generate P**H in A
				//              CWorkspace: need   2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zungbr('P', m, n, m, a, lda, work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				ldwkvt = (*m)
				if (*lwork) >= (*m)*(*n)+3*(*m) {
					//                 WORK( IVT ) is M by N
					nwork = ivt + ldwkvt*(*n)
					chunk = (*n)
				} else {
					//                 WORK( IVT ) is M by CHUNK
					chunk = ((*lwork) - 3*(*m)) / (*m)
					nwork = ivt + ldwkvt*chunk
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + BDSPAC
				Dbdsdc('L', 'I', m, s, rwork.Off(ie-1), rwork.MatrixOff(iru-1, *m, opts), m, rwork.MatrixOff(irvt-1, *m, opts), m, dum, &idum, rwork.Off(nrwork-1), iwork, info)

				//              Multiply Q in U by real matrix RWORK(IRVT)
				//              storing the result in WORK(IVT), copying to U
				//              CWorkspace: need   2*M [tauq, taup] + M*M [VT]
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + 2*M*M [rwork]
				Zlacrm(m, m, u, ldu, rwork.MatrixOff(iru-1, *m, opts), m, work.CMatrixOff(ivt-1, ldwkvt, opts), &ldwkvt, rwork.Off(nrwork-1))
				Zlacpy('F', m, m, work.CMatrixOff(ivt-1, ldwkvt, opts), &ldwkvt, u, ldu)

				//              Multiply RWORK(IRVT) by P**H in A, storing the
				//              result in WORK(IVT), copying to A
				//              CWorkspace: need   2*M [tauq, taup] + M*M [VT]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*N [VT]
				//              RWorkspace: need   M [e] + M*M [RVT] + 2*M*M [rwork]
				//              RWorkspace: prefer M [e] + M*M [RVT] + 2*M*N [rwork] < M + 5*M*M since N < 2*M here
				nrwork = iru
				for i = 1; i <= (*n); i += chunk {
					blk = minint((*n)-i+1, chunk)
					Zlarcm(m, &blk, rwork.MatrixOff(irvt-1, *m, opts), m, a.Off(0, i-1), lda, work.CMatrixOff(ivt-1, ldwkvt, opts), &ldwkvt, rwork.Off(nrwork-1))
					Zlacpy('F', m, &blk, work.CMatrixOff(ivt-1, ldwkvt, opts), &ldwkvt, a.Off(0, i-1), lda)
				}
			} else if wntqs {
				//              Path 5ts (N >> M, JOBZ='S')
				//              Copy A to U, generate Q
				//              CWorkspace: need   2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacpy('L', m, m, a, lda, u, ldu)
				Zungbr('Q', m, m, n, u, ldu, work.Off(itauq-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy A to VT, generate P**H
				//              CWorkspace: need   2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacpy('U', m, n, a, lda, vt, ldvt)
				Zungbr('P', m, n, m, vt, ldvt, work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + BDSPAC
				irvt = nrwork
				iru = irvt + (*m)*(*m)
				nrwork = iru + (*m)*(*m)
				Dbdsdc('L', 'I', m, s, rwork.Off(ie-1), rwork.MatrixOff(iru-1, *m, opts), m, rwork.MatrixOff(irvt-1, *m, opts), m, dum, &idum, rwork.Off(nrwork-1), iwork, info)

				//              Multiply Q in U by real matrix RWORK(IRU), storing the
				//              result in A, copying to U
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + 2*M*M [rwork]
				Zlacrm(m, m, u, ldu, rwork.MatrixOff(iru-1, *m, opts), m, a, lda, rwork.Off(nrwork-1))
				Zlacpy('F', m, m, a, lda, u, ldu)

				//              Multiply real matrix RWORK(IRVT) by P**H in VT,
				//              storing the result in A, copying to VT
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + 2*M*N [rwork] < M + 5*M*M since N < 2*M here
				nrwork = iru
				Zlarcm(m, n, rwork.MatrixOff(irvt-1, *m, opts), m, vt, ldvt, a, lda, rwork.Off(nrwork-1))
				Zlacpy('F', m, n, a, lda, vt, ldvt)
			} else {
				//              Path 5ta (N >> M, JOBZ='A')
				//              Copy A to U, generate Q
				//              CWorkspace: need   2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacpy('L', m, m, a, lda, u, ldu)
				Zungbr('Q', m, m, n, u, ldu, work.Off(itauq-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy A to VT, generate P**H
				//              CWorkspace: need   2*M [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacpy('U', m, n, a, lda, vt, ldvt)
				Zungbr('P', n, n, m, vt, ldvt, work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + BDSPAC
				irvt = nrwork
				iru = irvt + (*m)*(*m)
				nrwork = iru + (*m)*(*m)
				Dbdsdc('L', 'I', m, s, rwork.Off(ie-1), rwork.MatrixOff(iru-1, *m, opts), m, rwork.MatrixOff(irvt-1, *m, opts), m, dum, &idum, rwork.Off(nrwork-1), iwork, info)

				//              Multiply Q in U by real matrix RWORK(IRU), storing the
				//              result in A, copying to U
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + 2*M*M [rwork]
				Zlacrm(m, m, u, ldu, rwork.MatrixOff(iru-1, *m, opts), m, a, lda, rwork.Off(nrwork-1))
				Zlacpy('F', m, m, a, lda, u, ldu)

				//              Multiply real matrix RWORK(IRVT) by P**H in VT,
				//              storing the result in A, copying to VT
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + 2*M*N [rwork] < M + 5*M*M since N < 2*M here
				nrwork = iru
				Zlarcm(m, n, rwork.MatrixOff(irvt-1, *m, opts), m, vt, ldvt, a, lda, rwork.Off(nrwork-1))
				Zlacpy('F', m, n, a, lda, vt, ldvt)
			}

		} else {
			//           N .LT. MNTHR2
			//
			//           Path 6t (N > M, but not much larger)
			//           Reduce to bidiagonal form without LQ decomposition
			//           Use ZUNMBR to compute singular vectors
			ie = 1
			nrwork = ie + (*m)
			itauq = 1
			itaup = itauq + (*m)
			nwork = itaup + (*m)

			//           Bidiagonalize A
			//           CWorkspace: need   2*M [tauq, taup] + N        [work]
			//           CWorkspace: prefer 2*M [tauq, taup] + (M+N)*NB [work]
			//           RWorkspace: need   M [e]
			Zgebrd(m, n, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
			if wntqn {
				//              Path 6tn (N > M, JOBZ='N')
				//              Compute singular values only
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + BDSPAC
				Dbdsdc('L', 'N', m, s, rwork.Off(ie-1), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), dum, &idum, rwork.Off(nrwork-1), iwork, info)
			} else if wntqo {
				//              Path 6to (N > M, JOBZ='O')
				ldwkvt = (*m)
				ivt = nwork
				if (*lwork) >= (*m)*(*n)+3*(*m) {
					//                 WORK( IVT ) is M by N
					Zlaset('F', m, n, &czero, &czero, work.CMatrixOff(ivt-1, ldwkvt, opts), &ldwkvt)
					nwork = ivt + ldwkvt*(*n)
				} else {
					//                 WORK( IVT ) is M by CHUNK
					chunk = ((*lwork) - 3*(*m)) / (*m)
					nwork = ivt + ldwkvt*chunk
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + BDSPAC
				irvt = nrwork
				iru = irvt + (*m)*(*m)
				nrwork = iru + (*m)*(*m)
				Dbdsdc('L', 'I', m, s, rwork.Off(ie-1), rwork.MatrixOff(iru-1, *m, opts), m, rwork.MatrixOff(irvt-1, *m, opts), m, dum, &idum, rwork.Off(nrwork-1), iwork, info)

				//              Copy real matrix RWORK(IRU) to complex matrix U
				//              Overwrite U by left singular vectors of A
				//              CWorkspace: need   2*M [tauq, taup] + M*M [VT] + M    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*M [VT] + M*NB [work]
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU]
				Zlacp2('F', m, m, rwork.MatrixOff(iru-1, *m, opts), m, u, ldu)
				Zunmbr('Q', 'L', 'N', m, m, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				if (*lwork) >= (*m)*(*n)+3*(*m) {
					//                 Path 6to-fast
					//                 Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT)
					//                 Overwrite WORK(IVT) by right singular vectors of A,
					//                 copying to A
					//                 CWorkspace: need   2*M [tauq, taup] + M*N [VT] + M    [work]
					//                 CWorkspace: prefer 2*M [tauq, taup] + M*N [VT] + M*NB [work]
					//                 RWorkspace: need   M [e] + M*M [RVT]
					Zlacp2('F', m, m, rwork.MatrixOff(irvt-1, *m, opts), m, work.CMatrixOff(ivt-1, ldwkvt, opts), &ldwkvt)
					Zunmbr('P', 'R', 'C', m, n, m, a, lda, work.Off(itaup-1), work.CMatrixOff(ivt-1, ldwkvt, opts), &ldwkvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
					Zlacpy('F', m, n, work.CMatrixOff(ivt-1, ldwkvt, opts), &ldwkvt, a, lda)
				} else {
					//                 Path 6to-slow
					//                 Generate P**H in A
					//                 CWorkspace: need   2*M [tauq, taup] + M*M [VT] + M    [work]
					//                 CWorkspace: prefer 2*M [tauq, taup] + M*M [VT] + M*NB [work]
					//                 RWorkspace: need   0
					Zungbr('P', m, n, m, a, lda, work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

					//                 Multiply Q in A by real matrix RWORK(IRU), storing the
					//                 result in WORK(IU), copying to A
					//                 CWorkspace: need   2*M [tauq, taup] + M*M [VT]
					//                 CWorkspace: prefer 2*M [tauq, taup] + M*N [VT]
					//                 RWorkspace: need   M [e] + M*M [RVT] + 2*M*M [rwork]
					//                 RWorkspace: prefer M [e] + M*M [RVT] + 2*M*N [rwork] < M + 5*M*M since N < 2*M here
					nrwork = iru
					for i = 1; i <= (*n); i += chunk {
						blk = minint((*n)-i+1, chunk)
						Zlarcm(m, &blk, rwork.MatrixOff(irvt-1, *m, opts), m, a.Off(0, i-1), lda, work.CMatrixOff(ivt-1, ldwkvt, opts), &ldwkvt, rwork.Off(nrwork-1))
						Zlacpy('F', m, &blk, work.CMatrixOff(ivt-1, ldwkvt, opts), &ldwkvt, a.Off(0, i-1), lda)
					}
				}
			} else if wntqs {
				//              Path 6ts (N > M, JOBZ='S')
				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + BDSPAC
				irvt = nrwork
				iru = irvt + (*m)*(*m)
				nrwork = iru + (*m)*(*m)
				Dbdsdc('L', 'I', m, s, rwork.Off(ie-1), rwork.MatrixOff(iru-1, *m, opts), m, rwork.MatrixOff(irvt-1, *m, opts), m, dum, &idum, rwork.Off(nrwork-1), iwork, info)

				//              Copy real matrix RWORK(IRU) to complex matrix U
				//              Overwrite U by left singular vectors of A
				//              CWorkspace: need   2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU]
				Zlacp2('F', m, m, rwork.MatrixOff(iru-1, *m, opts), m, u, ldu)
				Zunmbr('Q', 'L', 'N', m, m, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy real matrix RWORK(IRVT) to complex matrix VT
				//              Overwrite VT by right singular vectors of A
				//              CWorkspace: need   2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   M [e] + M*M [RVT]
				Zlaset('F', m, n, &czero, &czero, vt, ldvt)
				Zlacp2('F', m, m, rwork.MatrixOff(irvt-1, *m, opts), m, vt, ldvt)
				Zunmbr('P', 'R', 'C', m, n, m, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
			} else {
				//              Path 6ta (N > M, JOBZ='A')
				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + BDSPAC
				irvt = nrwork
				iru = irvt + (*m)*(*m)
				nrwork = iru + (*m)*(*m)

				Dbdsdc('L', 'I', m, s, rwork.Off(ie-1), rwork.MatrixOff(iru-1, *m, opts), m, rwork.MatrixOff(irvt-1, *m, opts), m, dum, &idum, rwork.Off(nrwork-1), iwork, info)

				//              Copy real matrix RWORK(IRU) to complex matrix U
				//              Overwrite U by left singular vectors of A
				//              CWorkspace: need   2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU]
				Zlacp2('F', m, m, rwork.MatrixOff(iru-1, *m, opts), m, u, ldu)
				Zunmbr('Q', 'L', 'N', m, m, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Set all of VT to identity matrix
				Zlaset('F', n, n, &czero, &cone, vt, ldvt)

				//              Copy real matrix RWORK(IRVT) to complex matrix VT
				//              Overwrite VT by right singular vectors of A
				//              CWorkspace: need   2*M [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + N*NB [work]
				//              RWorkspace: need   M [e] + M*M [RVT]
				Zlacp2('F', m, m, rwork.MatrixOff(irvt-1, *m, opts), m, vt, ldvt)
				Zunmbr('P', 'R', 'C', n, n, m, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
			}

		}

	}

	//     Undo scaling if necessary
	if iscl == 1 {
		if anrm > bignum {
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bignum, &anrm, &minmn, func() *int { y := 1; return &y }(), s.Matrix(minmn, opts), &minmn, &ierr)
		}
		if (*info) != 0 && anrm > bignum {
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bignum, &anrm, toPtr(minmn-1), func() *int { y := 1; return &y }(), rwork.MatrixOff(ie-1, minmn, opts), &minmn, &ierr)
		}
		if anrm < smlnum {
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &smlnum, &anrm, &minmn, func() *int { y := 1; return &y }(), s.Matrix(minmn, opts), &minmn, &ierr)
		}
		if (*info) != 0 && anrm < smlnum {
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &smlnum, &anrm, toPtr(minmn-1), func() *int { y := 1; return &y }(), rwork.MatrixOff(ie-1, minmn, opts), &minmn, &ierr)
		}
	}

	//     Return optimal workspace in WORK(1)
	work.SetRe(0, float64(maxwrk))
}
