package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgesvd computes the singular value decomposition (SVD) of a complex
// M-by-N matrix A, optionally computing the left and/or right singular
// vectors. The SVD is written
//
//      A = U * SIGMA * conjugate-transpose(V)
//
// where SIGMA is an M-by-N matrix which is zero except for its
// min(m,n) diagonal elements, U is an M-by-M unitary matrix, and
// V is an N-by-N unitary matrix.  The diagonal elements of SIGMA
// are the singular values of A; they are real and non-negative, and
// are returned in descending order.  The first min(m,n) columns of
// U and V are the left and right singular vectors of A.
//
// Note that the routine returns V**H, not V.
func Zgesvd(jobu, jobvt byte, m, n *int, a *mat.CMatrix, lda *int, s *mat.Vector, u *mat.CMatrix, ldu *int, vt *mat.CMatrix, ldvt *int, work *mat.CVector, lwork *int, rwork *mat.Vector, info *int) {
	var lquery, wntua, wntuas, wntun, wntuo, wntus, wntva, wntvas, wntvn, wntvo, wntvs bool
	var cone, czero complex128
	var anrm, bignum, eps, one, smlnum, zero float64
	var blk, chunk, i, ie, ierr, ir, irwork, iscl, itau, itaup, itauq, iu, iwork, ldwrkr, ldwrku, lworkZgebrd, lworkZgelqf, lworkZgeqrf, lworkZungbrP, lworkZungbrQ, lworkZunglqM, lworkZunglqN, lworkZungqrM, lworkZungqrN, maxwrk, minmn, minwrk, mnthr, ncu, ncvt, nru, nrvt, wrkbl int
	var err error
	_ = err

	cdum := cvf(1)
	dum := vf(1)

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	zero = 0.0
	one = 1.0

	//     Test the input arguments
	(*info) = 0
	minmn = min(*m, *n)
	wntua = jobu == 'A'
	wntus = jobu == 'S'
	wntuas = wntua || wntus
	wntuo = jobu == 'O'
	wntun = jobu == 'N'
	wntva = jobvt == 'A'
	wntvs = jobvt == 'S'
	wntvas = wntva || wntvs
	wntvo = jobvt == 'O'
	wntvn = jobvt == 'N'
	lquery = ((*lwork) == -1)
	//
	if !(wntua || wntus || wntuo || wntun) {
		(*info) = -1
	} else if !(wntva || wntvs || wntvo || wntvn) || (wntvo && wntuo) {
		(*info) = -2
	} else if (*m) < 0 {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*lda) < max(1, *m) {
		(*info) = -6
	} else if (*ldu) < 1 || (wntuas && (*ldu) < (*m)) {
		(*info) = -9
	} else if (*ldvt) < 1 || (wntva && (*ldvt) < (*n)) || (wntvs && (*ldvt) < minmn) {
		(*info) = -11
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       CWorkspace refers to complex workspace, and RWorkspace to
	//       real workspace. NB refers to the optimal block size for the
	//       immediately following subroutine, as returned by ILAENV.)
	if (*info) == 0 {
		minwrk = 1
		maxwrk = 1
		if (*m) >= (*n) && minmn > 0 {
			//           Space needed for ZBDSQR is BDSPAC = 5*N
			mnthr = Ilaenv(func() *int { y := 6; return &y }(), []byte("ZGESVD"), []byte{jobu, jobvt}, m, n, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }())
			//           Compute space needed for ZGEQRF
			Zgeqrf(m, n, a, lda, cdum, cdum, toPtr(-1), &ierr)
			lworkZgeqrf = int(cdum.GetRe(0))
			//           Compute space needed for ZUNGQR
			Zungqr(m, n, n, a, lda, cdum, cdum, toPtr(-1), &ierr)
			lworkZungqrN = int(cdum.GetRe(0))
			Zungqr(m, m, n, a, lda, cdum, cdum, toPtr(-1), &ierr)
			lworkZungqrM = int(cdum.GetRe(0))
			//           Compute space needed for ZGEBRD
			Zgebrd(n, n, a, lda, s, dum.Off(0), cdum, cdum, cdum, toPtr(-1), &ierr)
			lworkZgebrd = int(cdum.GetRe(0))
			//           Compute space needed for ZUNGBR
			Zungbr('P', n, n, n, a, lda, cdum, cdum, toPtr(-1), &ierr)
			lworkZungbrP = int(cdum.GetRe(0))
			Zungbr('Q', n, n, n, a, lda, cdum, cdum, toPtr(-1), &ierr)
			lworkZungbrQ = int(cdum.GetRe(0))

			if (*m) >= mnthr {
				if wntun {
					//                 Path 1 (M much larger than N, JOBU='N')
					maxwrk = (*n) + lworkZgeqrf
					maxwrk = max(maxwrk, 2*(*n)+lworkZgebrd)
					if wntvo || wntvas {
						maxwrk = max(maxwrk, 2*(*n)+lworkZungbrP)
					}
					minwrk = 3 * (*n)
				} else if wntuo && wntvn {
					//                 Path 2 (M much larger than N, JOBU='O', JOBVT='N')
					wrkbl = (*n) + lworkZgeqrf
					wrkbl = max(wrkbl, (*n)+lworkZungqrN)
					wrkbl = max(wrkbl, 2*(*n)+lworkZgebrd)
					wrkbl = max(wrkbl, 2*(*n)+lworkZungbrQ)
					maxwrk = max((*n)*(*n)+wrkbl, (*n)*(*n)+(*m)*(*n))
					minwrk = 2*(*n) + (*m)
				} else if wntuo && wntvas {
					//                 Path 3 (M much larger than N, JOBU='O', JOBVT='S' or
					//                 'A')
					wrkbl = (*n) + lworkZgeqrf
					wrkbl = max(wrkbl, (*n)+lworkZungqrN)
					wrkbl = max(wrkbl, 2*(*n)+lworkZgebrd)
					wrkbl = max(wrkbl, 2*(*n)+lworkZungbrQ)
					wrkbl = max(wrkbl, 2*(*n)+lworkZungbrP)
					maxwrk = max((*n)*(*n)+wrkbl, (*n)*(*n)+(*m)*(*n))
					minwrk = 2*(*n) + (*m)
				} else if wntus && wntvn {
					//                 Path 4 (M much larger than N, JOBU='S', JOBVT='N')
					wrkbl = (*n) + lworkZgeqrf
					wrkbl = max(wrkbl, (*n)+lworkZungqrN)
					wrkbl = max(wrkbl, 2*(*n)+lworkZgebrd)
					wrkbl = max(wrkbl, 2*(*n)+lworkZungbrQ)
					maxwrk = (*n)*(*n) + wrkbl
					minwrk = 2*(*n) + (*m)
				} else if wntus && wntvo {
					//                 Path 5 (M much larger than N, JOBU='S', JOBVT='O')
					wrkbl = (*n) + lworkZgeqrf
					wrkbl = max(wrkbl, (*n)+lworkZungqrN)
					wrkbl = max(wrkbl, 2*(*n)+lworkZgebrd)
					wrkbl = max(wrkbl, 2*(*n)+lworkZungbrQ)
					wrkbl = max(wrkbl, 2*(*n)+lworkZungbrP)
					maxwrk = 2*(*n)*(*n) + wrkbl
					minwrk = 2*(*n) + (*m)
				} else if wntus && wntvas {
					//                 Path 6 (M much larger than N, JOBU='S', JOBVT='S' or
					//                 'A')
					wrkbl = (*n) + lworkZgeqrf
					wrkbl = max(wrkbl, (*n)+lworkZungqrN)
					wrkbl = max(wrkbl, 2*(*n)+lworkZgebrd)
					wrkbl = max(wrkbl, 2*(*n)+lworkZungbrQ)
					wrkbl = max(wrkbl, 2*(*n)+lworkZungbrP)
					maxwrk = (*n)*(*n) + wrkbl
					minwrk = 2*(*n) + (*m)
				} else if wntua && wntvn {
					//                 Path 7 (M much larger than N, JOBU='A', JOBVT='N')
					wrkbl = (*n) + lworkZgeqrf
					wrkbl = max(wrkbl, (*n)+lworkZungqrM)
					wrkbl = max(wrkbl, 2*(*n)+lworkZgebrd)
					wrkbl = max(wrkbl, 2*(*n)+lworkZungbrQ)
					maxwrk = (*n)*(*n) + wrkbl
					minwrk = 2*(*n) + (*m)
				} else if wntua && wntvo {
					//                 Path 8 (M much larger than N, JOBU='A', JOBVT='O')
					wrkbl = (*n) + lworkZgeqrf
					wrkbl = max(wrkbl, (*n)+lworkZungqrM)
					wrkbl = max(wrkbl, 2*(*n)+lworkZgebrd)
					wrkbl = max(wrkbl, 2*(*n)+lworkZungbrQ)
					wrkbl = max(wrkbl, 2*(*n)+lworkZungbrP)
					maxwrk = 2*(*n)*(*n) + wrkbl
					minwrk = 2*(*n) + (*m)
				} else if wntua && wntvas {
					//                 Path 9 (M much larger than N, JOBU='A', JOBVT='S' or
					//                 'A')
					wrkbl = (*n) + lworkZgeqrf
					wrkbl = max(wrkbl, (*n)+lworkZungqrM)
					wrkbl = max(wrkbl, 2*(*n)+lworkZgebrd)
					wrkbl = max(wrkbl, 2*(*n)+lworkZungbrQ)
					wrkbl = max(wrkbl, 2*(*n)+lworkZungbrP)
					maxwrk = (*n)*(*n) + wrkbl
					minwrk = 2*(*n) + (*m)
				}
			} else {
				//              Path 10 (M at least N, but not much larger)
				Zgebrd(m, n, a, lda, s, dum.Off(0), cdum, cdum, cdum, toPtr(-1), &ierr)
				lworkZgebrd = int(cdum.GetRe(0))
				maxwrk = 2*(*n) + lworkZgebrd
				if wntus || wntuo {
					Zungbr('Q', m, n, n, a, lda, cdum, cdum, toPtr(-1), &ierr)
					lworkZungbrQ = int(cdum.GetRe(0))
					maxwrk = max(maxwrk, 2*(*n)+lworkZungbrQ)
				}
				if wntua {
					Zungbr('Q', m, m, n, a, lda, cdum, cdum, toPtr(-1), &ierr)
					lworkZungbrQ = int(cdum.GetRe(0))
					maxwrk = max(maxwrk, 2*(*n)+lworkZungbrQ)
				}
				if !wntvn {
					maxwrk = max(maxwrk, 2*(*n)+lworkZungbrP)
				}
				minwrk = 2*(*n) + (*m)
			}
		} else if minmn > 0 {
			//           Space needed for ZBDSQR is BDSPAC = 5*M
			mnthr = Ilaenv(func() *int { y := 6; return &y }(), []byte("ZGESVD"), []byte{jobu, jobvt}, m, n, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }())
			//           Compute space needed for ZGELQF
			Zgelqf(m, n, a, lda, cdum, cdum, toPtr(-1), &ierr)
			lworkZgelqf = int(cdum.GetRe(0))
			//           Compute space needed for ZUNGLQ
			Zunglq(n, n, m, cdum.CMatrix(*n, opts), n, cdum, cdum, toPtr(-1), &ierr)
			lworkZunglqN = int(cdum.GetRe(0))
			Zunglq(m, n, m, a, lda, cdum, cdum, toPtr(-1), &ierr)
			lworkZunglqM = int(cdum.GetRe(0))
			//           Compute space needed for ZGEBRD
			Zgebrd(m, m, a, lda, s, dum.Off(0), cdum, cdum, cdum, toPtr(-1), &ierr)
			lworkZgebrd = int(cdum.GetRe(0))
			//            Compute space needed for ZUNGBR P
			Zungbr('P', m, m, m, a, n, cdum, cdum, toPtr(-1), &ierr)
			lworkZungbrP = int(cdum.GetRe(0))
			//           Compute space needed for ZUNGBR Q
			Zungbr('Q', m, m, m, a, n, cdum, cdum, toPtr(-1), &ierr)
			lworkZungbrQ = int(cdum.GetRe(0))
			if (*n) >= mnthr {
				if wntvn {
					//                 Path 1t(N much larger than M, JOBVT='N')
					maxwrk = (*m) + lworkZgelqf
					maxwrk = max(maxwrk, 2*(*m)+lworkZgebrd)
					if wntuo || wntuas {
						maxwrk = max(maxwrk, 2*(*m)+lworkZungbrQ)
					}
					minwrk = 3 * (*m)
				} else if wntvo && wntun {
					//                 Path 2t(N much larger than M, JOBU='N', JOBVT='O')
					wrkbl = (*m) + lworkZgelqf
					wrkbl = max(wrkbl, (*m)+lworkZunglqM)
					wrkbl = max(wrkbl, 2*(*m)+lworkZgebrd)
					wrkbl = max(wrkbl, 2*(*m)+lworkZungbrP)
					maxwrk = max((*m)*(*m)+wrkbl, (*m)*(*m)+(*m)*(*n))
					minwrk = 2*(*m) + (*n)
				} else if wntvo && wntuas {
					//                 Path 3t(N much larger than M, JOBU='S' or 'A',
					//                 JOBVT='O')
					wrkbl = (*m) + lworkZgelqf
					wrkbl = max(wrkbl, (*m)+lworkZunglqM)
					wrkbl = max(wrkbl, 2*(*m)+lworkZgebrd)
					wrkbl = max(wrkbl, 2*(*m)+lworkZungbrP)
					wrkbl = max(wrkbl, 2*(*m)+lworkZungbrQ)
					maxwrk = max((*m)*(*m)+wrkbl, (*m)*(*m)+(*m)*(*n))
					minwrk = 2*(*m) + (*n)
				} else if wntvs && wntun {
					//                 Path 4t(N much larger than M, JOBU='N', JOBVT='S')
					wrkbl = (*m) + lworkZgelqf
					wrkbl = max(wrkbl, (*m)+lworkZunglqM)
					wrkbl = max(wrkbl, 2*(*m)+lworkZgebrd)
					wrkbl = max(wrkbl, 2*(*m)+lworkZungbrP)
					maxwrk = (*m)*(*m) + wrkbl
					minwrk = 2*(*m) + (*n)
				} else if wntvs && wntuo {
					//                 Path 5t(N much larger than M, JOBU='O', JOBVT='S')
					wrkbl = (*m) + lworkZgelqf
					wrkbl = max(wrkbl, (*m)+lworkZunglqM)
					wrkbl = max(wrkbl, 2*(*m)+lworkZgebrd)
					wrkbl = max(wrkbl, 2*(*m)+lworkZungbrP)
					wrkbl = max(wrkbl, 2*(*m)+lworkZungbrQ)
					maxwrk = 2*(*m)*(*m) + wrkbl
					minwrk = 2*(*m) + (*n)
				} else if wntvs && wntuas {
					//                 Path 6t(N much larger than M, JOBU='S' or 'A',
					//                 JOBVT='S')
					wrkbl = (*m) + lworkZgelqf
					wrkbl = max(wrkbl, (*m)+lworkZunglqM)
					wrkbl = max(wrkbl, 2*(*m)+lworkZgebrd)
					wrkbl = max(wrkbl, 2*(*m)+lworkZungbrP)
					wrkbl = max(wrkbl, 2*(*m)+lworkZungbrQ)
					maxwrk = (*m)*(*m) + wrkbl
					minwrk = 2*(*m) + (*n)
				} else if wntva && wntun {
					//                 Path 7t(N much larger than M, JOBU='N', JOBVT='A')
					wrkbl = (*m) + lworkZgelqf
					wrkbl = max(wrkbl, (*m)+lworkZunglqN)
					wrkbl = max(wrkbl, 2*(*m)+lworkZgebrd)
					wrkbl = max(wrkbl, 2*(*m)+lworkZungbrP)
					maxwrk = (*m)*(*m) + wrkbl
					minwrk = 2*(*m) + (*n)
				} else if wntva && wntuo {
					//                 Path 8t(N much larger than M, JOBU='O', JOBVT='A')
					wrkbl = (*m) + lworkZgelqf
					wrkbl = max(wrkbl, (*m)+lworkZunglqN)
					wrkbl = max(wrkbl, 2*(*m)+lworkZgebrd)
					wrkbl = max(wrkbl, 2*(*m)+lworkZungbrP)
					wrkbl = max(wrkbl, 2*(*m)+lworkZungbrQ)
					maxwrk = 2*(*m)*(*m) + wrkbl
					minwrk = 2*(*m) + (*n)
				} else if wntva && wntuas {
					//                 Path 9t(N much larger than M, JOBU='S' or 'A',
					//                 JOBVT='A')
					wrkbl = (*m) + lworkZgelqf
					wrkbl = max(wrkbl, (*m)+lworkZunglqN)
					wrkbl = max(wrkbl, 2*(*m)+lworkZgebrd)
					wrkbl = max(wrkbl, 2*(*m)+lworkZungbrP)
					wrkbl = max(wrkbl, 2*(*m)+lworkZungbrQ)
					maxwrk = (*m)*(*m) + wrkbl
					minwrk = 2*(*m) + (*n)
				}
			} else {
				//              Path 10t(N greater than M, but not much larger)
				Zgebrd(m, n, a, lda, s, dum.Off(0), cdum, cdum, cdum, toPtr(-1), &ierr)
				lworkZgebrd = int(cdum.GetRe(0))
				maxwrk = 2*(*m) + lworkZgebrd
				if wntvs || wntvo {
					//                Compute space needed for ZUNGBR P
					Zungbr('P', m, n, m, a, n, cdum, cdum, toPtr(-1), &ierr)
					lworkZungbrP = int(cdum.GetRe(0))
					maxwrk = max(maxwrk, 2*(*m)+lworkZungbrP)
				}
				if wntva {
					Zungbr('P', n, n, m, a, n, cdum, cdum, toPtr(-1), &ierr)
					lworkZungbrP = int(cdum.GetRe(0))
					maxwrk = max(maxwrk, 2*(*m)+lworkZungbrP)
				}
				if !wntun {
					maxwrk = max(maxwrk, 2*(*m)+lworkZungbrQ)
				}
				minwrk = 2*(*m) + (*n)
			}
		}
		maxwrk = max(maxwrk, minwrk)
		work.SetRe(0, float64(maxwrk))

		if (*lwork) < minwrk && !lquery {
			(*info) = -13
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGESVD"), -(*info))
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

	//     Scale A if max element outside range [SMLNUM,BIGNUM]
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
		if (*m) >= mnthr {

			if wntun {
				//              Path 1 (M much larger than N, JOBU='N')
				//              No left singular vectors to be computed
				itau = 1
				iwork = itau + (*n)

				//              Compute A=Q*R
				//              (CWorkspace: need 2*N, prefer N+N*NB)
				//              (RWorkspace: need 0)
				Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

				//              Zero out below R
				if (*n) > 1 {
					Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, a.Off(1, 0), lda)
				}
				ie = 1
				itauq = 1
				itaup = itauq + (*n)
				iwork = itaup + (*n)

				//              Bidiagonalize R in A
				//              (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
				//              (RWorkspace: need N)
				Zgebrd(n, n, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
				ncvt = 0
				if wntvo || wntvas {
					//                 If right singular vectors desired, generate P'.
					//                 (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
					//                 (RWorkspace: 0)
					Zungbr('P', n, n, n, a, lda, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					ncvt = (*n)
				}
				irwork = ie + (*n)

				//              Perform bidiagonal QR iteration, computing right
				//              singular vectors of A in A if desired
				//              (CWorkspace: 0)
				//              (RWorkspace: need BDSPAC)
				Zbdsqr('U', n, &ncvt, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), a, lda, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

				//              If right singular vectors desired in VT, copy them there
				if wntvas {
					Zlacpy('F', n, n, a, lda, vt, ldvt)
				}

			} else if wntuo && wntvn {
				//              Path 2 (M much larger than N, JOBU='O', JOBVT='N')
				//              N left singular vectors to be overwritten on A and
				//              no right singular vectors to be computed
				if (*lwork) >= (*n)*(*n)+3*(*n) {
					//                 Sufficient workspace for a fast algorithm
					ir = 1
					if (*lwork) >= max(wrkbl, (*lda)*(*n))+(*lda)*(*n) {
						//                    WORK(IU) is LDA by N, WORK(IR) is LDA by N
						ldwrku = (*lda)
						ldwrkr = (*lda)
					} else if (*lwork) >= max(wrkbl, (*lda)*(*n))+(*n)*(*n) {
						//                    WORK(IU) is LDA by N, WORK(IR) is N by N
						ldwrku = (*lda)
						ldwrkr = (*n)
					} else {
						//                    WORK(IU) is LDWRKU by N, WORK(IR) is N by N
						ldwrku = ((*lwork) - (*n)*(*n)) / (*n)
						ldwrkr = (*n)
					}
					itau = ir + ldwrkr*(*n)
					iwork = itau + (*n)

					//                 Compute A=Q*R
					//                 (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
					//                 (RWorkspace: 0)
					Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Copy R to WORK(IR) and zero out below it
					Zlacpy('U', n, n, a, lda, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr)
					Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, work.CMatrixOff(ir, ldwrkr, opts), &ldwrkr)

					//                 Generate Q in A
					//                 (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
					//                 (RWorkspace: 0)
					Zungqr(m, n, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					ie = 1
					itauq = itau
					itaup = itauq + (*n)
					iwork = itaup + (*n)

					//                 Bidiagonalize R in WORK(IR)
					//                 (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
					//                 (RWorkspace: need N)
					Zgebrd(n, n, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Generate left vectors bidiagonalizing R
					//                 (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
					//                 (RWorkspace: need 0)
					Zungbr('Q', n, n, n, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					irwork = ie + (*n)

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of R in WORK(IR)
					//                 (CWorkspace: need N*N)
					//                 (RWorkspace: need BDSPAC)
					Zbdsqr('U', n, func() *int { y := 0; return &y }(), n, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)
					iu = itauq

					//                 Multiply Q in A by left singular vectors of R in
					//                 WORK(IR), storing result in WORK(IU) and copying to A
					//                 (CWorkspace: need N*N+N, prefer N*N+M*N)
					//                 (RWorkspace: 0)
					for i = 1; i <= (*m); i += ldwrku {
						chunk = min((*m)-i+1, ldwrku)
						err = goblas.Zgemm(NoTrans, NoTrans, chunk, *n, *n, cone, a.Off(i-1, 0), work.CMatrixOff(ir-1, ldwrkr, opts), czero, work.CMatrixOff(iu-1, ldwrku, opts))
						Zlacpy('F', &chunk, n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, a.Off(i-1, 0), lda)
					}

				} else {
					//                 Insufficient workspace for a fast algorithm
					ie = 1
					itauq = 1
					itaup = itauq + (*n)
					iwork = itaup + (*n)

					//                 Bidiagonalize A
					//                 (CWorkspace: need 2*N+M, prefer 2*N+(M+N)*NB)
					//                 (RWorkspace: N)
					Zgebrd(m, n, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Generate left vectors bidiagonalizing A
					//                 (CWorkspace: need 3*N, prefer 2*N+N*NB)
					//                 (RWorkspace: 0)
					Zungbr('Q', m, n, n, a, lda, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					irwork = ie + (*n)

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of A in A
					//                 (CWorkspace: need 0)
					//                 (RWorkspace: need BDSPAC)
					Zbdsqr('U', n, func() *int { y := 0; return &y }(), m, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), a, lda, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

				}

			} else if wntuo && wntvas {
				//              Path 3 (M much larger than N, JOBU='O', JOBVT='S' or 'A')
				//              N left singular vectors to be overwritten on A and
				//              N right singular vectors to be computed in VT
				if (*lwork) >= (*n)*(*n)+3*(*n) {
					//                 Sufficient workspace for a fast algorithm
					ir = 1
					if (*lwork) >= max(wrkbl, (*lda)*(*n))+(*lda)*(*n) {
						//                    WORK(IU) is LDA by N and WORK(IR) is LDA by N
						ldwrku = (*lda)
						ldwrkr = (*lda)
					} else if (*lwork) >= max(wrkbl, (*lda)*(*n))+(*n)*(*n) {
						//                    WORK(IU) is LDA by N and WORK(IR) is N by N
						ldwrku = (*lda)
						ldwrkr = (*n)
					} else {
						//                    WORK(IU) is LDWRKU by N and WORK(IR) is N by N
						ldwrku = ((*lwork) - (*n)*(*n)) / (*n)
						ldwrkr = (*n)
					}
					itau = ir + ldwrkr*(*n)
					iwork = itau + (*n)

					//                 Compute A=Q*R
					//                 (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
					//                 (RWorkspace: 0)
					Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Copy R to VT, zeroing out below it
					Zlacpy('U', n, n, a, lda, vt, ldvt)
					if (*n) > 1 {
						Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, vt.Off(1, 0), ldvt)
					}

					//                 Generate Q in A
					//                 (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
					//                 (RWorkspace: 0)
					Zungqr(m, n, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					ie = 1
					itauq = itau
					itaup = itauq + (*n)
					iwork = itaup + (*n)

					//                 Bidiagonalize R in VT, copying result to WORK(IR)
					//                 (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
					//                 (RWorkspace: need N)
					Zgebrd(n, n, vt, ldvt, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					Zlacpy('L', n, n, vt, ldvt, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr)

					//                 Generate left vectors bidiagonalizing R in WORK(IR)
					//                 (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
					//                 (RWorkspace: 0)
					Zungbr('Q', n, n, n, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Generate right vectors bidiagonalizing R in VT
					//                 (CWorkspace: need N*N+3*N-1, prefer N*N+2*N+(N-1)*NB)
					//                 (RWorkspace: 0)
					Zungbr('P', n, n, n, vt, ldvt, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					irwork = ie + (*n)

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of R in WORK(IR) and computing right
					//                 singular vectors of R in VT
					//                 (CWorkspace: need N*N)
					//                 (RWorkspace: need BDSPAC)
					Zbdsqr('U', n, n, n, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), vt, ldvt, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)
					iu = itauq

					//                 Multiply Q in A by left singular vectors of R in
					//                 WORK(IR), storing result in WORK(IU) and copying to A
					//                 (CWorkspace: need N*N+N, prefer N*N+M*N)
					//                 (RWorkspace: 0)
					for i = 1; i <= (*m); i += ldwrku {
						chunk = min((*m)-i+1, ldwrku)
						err = goblas.Zgemm(NoTrans, NoTrans, chunk, *n, *n, cone, a.Off(i-1, 0), work.CMatrixOff(ir-1, ldwrkr, opts), czero, work.CMatrixOff(iu-1, ldwrku, opts))
						Zlacpy('F', &chunk, n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, a.Off(i-1, 0), lda)
					}

				} else {
					//                 Insufficient workspace for a fast algorithm
					itau = 1
					iwork = itau + (*n)

					//                 Compute A=Q*R
					//                 (CWorkspace: need 2*N, prefer N+N*NB)
					//                 (RWorkspace: 0)
					Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Copy R to VT, zeroing out below it
					Zlacpy('U', n, n, a, lda, vt, ldvt)
					if (*n) > 1 {
						Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, vt.Off(1, 0), ldvt)
					}

					//                 Generate Q in A
					//                 (CWorkspace: need 2*N, prefer N+N*NB)
					//                 (RWorkspace: 0)
					Zungqr(m, n, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					ie = 1
					itauq = itau
					itaup = itauq + (*n)
					iwork = itaup + (*n)

					//                 Bidiagonalize R in VT
					//                 (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
					//                 (RWorkspace: N)
					Zgebrd(n, n, vt, ldvt, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Multiply Q in A by left vectors bidiagonalizing R
					//                 (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
					//                 (RWorkspace: 0)
					Zunmbr('Q', 'R', 'N', m, n, n, vt, ldvt, work.Off(itauq-1), a, lda, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Generate right vectors bidiagonalizing R in VT
					//                 (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
					//                 (RWorkspace: 0)
					Zungbr('P', n, n, n, vt, ldvt, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					irwork = ie + (*n)

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of A in A and computing right
					//                 singular vectors of A in VT
					//                 (CWorkspace: 0)
					//                 (RWorkspace: need BDSPAC)
					Zbdsqr('U', n, n, m, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), vt, ldvt, a, lda, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

				}

			} else if wntus {

				if wntvn {
					//                 Path 4 (M much larger than N, JOBU='S', JOBVT='N')
					//                 N left singular vectors to be computed in U and
					//                 no right singular vectors to be computed
					if (*lwork) >= (*n)*(*n)+3*(*n) {
						//                    Sufficient workspace for a fast algorithm
						ir = 1
						if (*lwork) >= wrkbl+(*lda)*(*n) {
							//                       WORK(IR) is LDA by N
							ldwrkr = (*lda)
						} else {
							//                       WORK(IR) is N by N
							ldwrkr = (*n)
						}
						itau = ir + ldwrkr*(*n)
						iwork = itau + (*n)

						//                    Compute A=Q*R
						//                    (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
						//                    (RWorkspace: 0)
						Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy R to WORK(IR), zeroing out below it
						Zlacpy('U', n, n, a, lda, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr)
						Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, work.CMatrixOff(ir, ldwrkr, opts), &ldwrkr)

						//                    Generate Q in A
						//                    (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
						//                    (RWorkspace: 0)
						Zungqr(m, n, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = 1
						itauq = itau
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Bidiagonalize R in WORK(IR)
						//                    (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
						//                    (RWorkspace: need N)
						Zgebrd(n, n, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left vectors bidiagonalizing R in WORK(IR)
						//                    (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
						//                    (RWorkspace: 0)
						Zungbr('Q', n, n, n, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IR)
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', n, func() *int { y := 0; return &y }(), n, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

						//                    Multiply Q in A by left singular vectors of R in
						//                    WORK(IR), storing result in U
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: 0)
						err = goblas.Zgemm(NoTrans, NoTrans, *m, *n, *n, cone, a, work.CMatrixOff(ir-1, ldwrkr, opts), czero, u)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*n)

						//                    Compute A=Q*R, copying result to U
						//                    (CWorkspace: need 2*N, prefer N+N*NB)
						//                    (RWorkspace: 0)
						Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('L', m, n, a, lda, u, ldu)

						//                    Generate Q in U
						//                    (CWorkspace: need 2*N, prefer N+N*NB)
						//                    (RWorkspace: 0)
						Zungqr(m, n, n, u, ldu, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = 1
						itauq = itau
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Zero out below R in A
						if (*n) > 1 {
							Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, a.Off(1, 0), lda)
						}

						//                    Bidiagonalize R in A
						//                    (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
						//                    (RWorkspace: need N)
						Zgebrd(n, n, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply Q in U by left vectors bidiagonalizing R
						//                    (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
						//                    (RWorkspace: 0)
						Zunmbr('Q', 'R', 'N', m, n, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', n, func() *int { y := 0; return &y }(), m, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), u, ldu, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

					}

				} else if wntvo {
					//                 Path 5 (M much larger than N, JOBU='S', JOBVT='O')
					//                 N left singular vectors to be computed in U and
					//                 N right singular vectors to be overwritten on A
					if (*lwork) >= 2*(*n)*(*n)+3*(*n) {
						//                    Sufficient workspace for a fast algorithm
						iu = 1
						if (*lwork) >= wrkbl+2*(*lda)*(*n) {
							//                       WORK(IU) is LDA by N and WORK(IR) is LDA by N
							ldwrku = (*lda)
							ir = iu + ldwrku*(*n)
							ldwrkr = (*lda)
						} else if (*lwork) >= wrkbl+((*lda)+(*n))*(*n) {
							//                       WORK(IU) is LDA by N and WORK(IR) is N by N
							ldwrku = (*lda)
							ir = iu + ldwrku*(*n)
							ldwrkr = (*n)
						} else {
							//                       WORK(IU) is N by N and WORK(IR) is N by N
							ldwrku = (*n)
							ir = iu + ldwrku*(*n)
							ldwrkr = (*n)
						}
						itau = ir + ldwrkr*(*n)
						iwork = itau + (*n)

						//                    Compute A=Q*R
						//                    (CWorkspace: need 2*N*N+2*N, prefer 2*N*N+N+N*NB)
						//                    (RWorkspace: 0)
						Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy R to WORK(IU), zeroing out below it
						Zlacpy('U', n, n, a, lda, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, work.CMatrixOff(iu, ldwrku, opts), &ldwrku)

						//                    Generate Q in A
						//                    (CWorkspace: need 2*N*N+2*N, prefer 2*N*N+N+N*NB)
						//                    (RWorkspace: 0)
						Zungqr(m, n, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = 1
						itauq = itau
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Bidiagonalize R in WORK(IU), copying result to
						//                    WORK(IR)
						//                    (CWorkspace: need   2*N*N+3*N,
						//                                 prefer 2*N*N+2*N+2*N*NB)
						//                    (RWorkspace: need   N)
						Zgebrd(n, n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('U', n, n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr)

						//                    Generate left bidiagonalizing vectors in WORK(IU)
						//                    (CWorkspace: need 2*N*N+3*N, prefer 2*N*N+2*N+N*NB)
						//                    (RWorkspace: 0)
						Zungbr('Q', n, n, n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right bidiagonalizing vectors in WORK(IR)
						//                    (CWorkspace: need   2*N*N+3*N-1,
						//                                 prefer 2*N*N+2*N+(N-1)*NB)
						//                    (RWorkspace: 0)
						Zungbr('P', n, n, n, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IU) and computing
						//                    right singular vectors of R in WORK(IR)
						//                    (CWorkspace: need 2*N*N)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', n, n, n, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

						//                    Multiply Q in A by left singular vectors of R in
						//                    WORK(IU), storing result in U
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: 0)
						err = goblas.Zgemm(NoTrans, NoTrans, *m, *n, *n, cone, a, work.CMatrixOff(iu-1, ldwrku, opts), czero, u)

						//                    Copy right singular vectors of R to A
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: 0)
						Zlacpy('F', n, n, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, a, lda)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*n)

						//                    Compute A=Q*R, copying result to U
						//                    (CWorkspace: need 2*N, prefer N+N*NB)
						//                    (RWorkspace: 0)
						Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('L', m, n, a, lda, u, ldu)

						//                    Generate Q in U
						//                    (CWorkspace: need 2*N, prefer N+N*NB)
						//                    (RWorkspace: 0)
						Zungqr(m, n, n, u, ldu, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = 1
						itauq = itau
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Zero out below R in A
						if (*n) > 1 {
							Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, a.Off(1, 0), lda)
						}

						//                    Bidiagonalize R in A
						//                    (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
						//                    (RWorkspace: need N)
						Zgebrd(n, n, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply Q in U by left vectors bidiagonalizing R
						//                    (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
						//                    (RWorkspace: 0)
						Zunmbr('Q', 'R', 'N', m, n, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right vectors bidiagonalizing R in A
						//                    (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
						//                    (RWorkspace: 0)
						Zungbr('P', n, n, n, a, lda, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in A
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', n, n, m, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), a, lda, u, ldu, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

					}

				} else if wntvas {
					//                 Path 6 (M much larger than N, JOBU='S', JOBVT='S'
					//                         or 'A')
					//                 N left singular vectors to be computed in U and
					//                 N right singular vectors to be computed in VT
					if (*lwork) >= (*n)*(*n)+3*(*n) {
						//                    Sufficient workspace for a fast algorithm
						iu = 1
						if (*lwork) >= wrkbl+(*lda)*(*n) {
							//                       WORK(IU) is LDA by N
							ldwrku = (*lda)
						} else {
							//                       WORK(IU) is N by N
							ldwrku = (*n)
						}
						itau = iu + ldwrku*(*n)
						iwork = itau + (*n)

						//                    Compute A=Q*R
						//                    (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
						//                    (RWorkspace: 0)
						Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy R to WORK(IU), zeroing out below it
						Zlacpy('U', n, n, a, lda, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, work.CMatrixOff(iu, ldwrku, opts), &ldwrku)

						//                    Generate Q in A
						//                    (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
						//                    (RWorkspace: 0)
						Zungqr(m, n, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = 1
						itauq = itau
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Bidiagonalize R in WORK(IU), copying result to VT
						//                    (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
						//                    (RWorkspace: need N)
						Zgebrd(n, n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('U', n, n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, vt, ldvt)

						//                    Generate left bidiagonalizing vectors in WORK(IU)
						//                    (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
						//                    (RWorkspace: 0)
						Zungbr('Q', n, n, n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right bidiagonalizing vectors in VT
						//                    (CWorkspace: need   N*N+3*N-1,
						//                                 prefer N*N+2*N+(N-1)*NB)
						//                    (RWorkspace: 0)
						Zungbr('P', n, n, n, vt, ldvt, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IU) and computing
						//                    right singular vectors of R in VT
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', n, n, n, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), vt, ldvt, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

						//                    Multiply Q in A by left singular vectors of R in
						//                    WORK(IU), storing result in U
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: 0)
						err = goblas.Zgemm(NoTrans, NoTrans, *m, *n, *n, cone, a, work.CMatrixOff(iu-1, ldwrku, opts), czero, u)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*n)

						//                    Compute A=Q*R, copying result to U
						//                    (CWorkspace: need 2*N, prefer N+N*NB)
						//                    (RWorkspace: 0)
						Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('L', m, n, a, lda, u, ldu)

						//                    Generate Q in U
						//                    (CWorkspace: need 2*N, prefer N+N*NB)
						//                    (RWorkspace: 0)
						Zungqr(m, n, n, u, ldu, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy R to VT, zeroing out below it
						Zlacpy('U', n, n, a, lda, vt, ldvt)
						if (*n) > 1 {
							Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, vt.Off(1, 0), ldvt)
						}
						ie = 1
						itauq = itau
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Bidiagonalize R in VT
						//                    (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
						//                    (RWorkspace: need N)
						Zgebrd(n, n, vt, ldvt, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply Q in U by left bidiagonalizing vectors
						//                    in VT
						//                    (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
						//                    (RWorkspace: 0)
						Zunmbr('Q', 'R', 'N', m, n, n, vt, ldvt, work.Off(itauq-1), u, ldu, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right bidiagonalizing vectors in VT
						//                    (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
						//                    (RWorkspace: 0)
						Zungbr('P', n, n, n, vt, ldvt, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in VT
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', n, n, m, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), vt, ldvt, u, ldu, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

					}

				}

			} else if wntua {

				if wntvn {
					//                 Path 7 (M much larger than N, JOBU='A', JOBVT='N')
					//                 M left singular vectors to be computed in U and
					//                 no right singular vectors to be computed
					if (*lwork) >= (*n)*(*n)+max((*n)+(*m), 3*(*n)) {
						//                    Sufficient workspace for a fast algorithm
						ir = 1
						if (*lwork) >= wrkbl+(*lda)*(*n) {
							//                       WORK(IR) is LDA by N
							ldwrkr = (*lda)
						} else {
							//                       WORK(IR) is N by N
							ldwrkr = (*n)
						}
						itau = ir + ldwrkr*(*n)
						iwork = itau + (*n)

						//                    Compute A=Q*R, copying result to U
						//                    (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
						//                    (RWorkspace: 0)
						Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('L', m, n, a, lda, u, ldu)

						//                    Copy R to WORK(IR), zeroing out below it
						Zlacpy('U', n, n, a, lda, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr)
						Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, work.CMatrixOff(ir, ldwrkr, opts), &ldwrkr)

						//                    Generate Q in U
						//                    (CWorkspace: need N*N+N+M, prefer N*N+N+M*NB)
						//                    (RWorkspace: 0)
						Zungqr(m, m, n, u, ldu, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = 1
						itauq = itau
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Bidiagonalize R in WORK(IR)
						//                    (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
						//                    (RWorkspace: need N)
						Zgebrd(n, n, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left bidiagonalizing vectors in WORK(IR)
						//                    (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
						//                    (RWorkspace: 0)
						Zungbr('Q', n, n, n, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IR)
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', n, func() *int { y := 0; return &y }(), n, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

						//                    Multiply Q in U by left singular vectors of R in
						//                    WORK(IR), storing result in A
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: 0)
						err = goblas.Zgemm(NoTrans, NoTrans, *m, *n, *n, cone, u, work.CMatrixOff(ir-1, ldwrkr, opts), czero, a)

						//                    Copy left singular vectors of A from A to U
						Zlacpy('F', m, n, a, lda, u, ldu)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*n)

						//                    Compute A=Q*R, copying result to U
						//                    (CWorkspace: need 2*N, prefer N+N*NB)
						//                    (RWorkspace: 0)
						Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('L', m, n, a, lda, u, ldu)

						//                    Generate Q in U
						//                    (CWorkspace: need N+M, prefer N+M*NB)
						//                    (RWorkspace: 0)
						Zungqr(m, m, n, u, ldu, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = 1
						itauq = itau
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Zero out below R in A
						if (*n) > 1 {
							Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, a.Off(1, 0), lda)
						}

						//                    Bidiagonalize R in A
						//                    (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
						//                    (RWorkspace: need N)
						Zgebrd(n, n, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply Q in U by left bidiagonalizing vectors
						//                    in A
						//                    (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
						//                    (RWorkspace: 0)
						Zunmbr('Q', 'R', 'N', m, n, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', n, func() *int { y := 0; return &y }(), m, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), u, ldu, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

					}

				} else if wntvo {
					//                 Path 8 (M much larger than N, JOBU='A', JOBVT='O')
					//                 M left singular vectors to be computed in U and
					//                 N right singular vectors to be overwritten on A
					if (*lwork) >= 2*(*n)*(*n)+max((*n)+(*m), 3*(*n)) {
						//                    Sufficient workspace for a fast algorithm
						iu = 1
						if (*lwork) >= wrkbl+2*(*lda)*(*n) {
							//                       WORK(IU) is LDA by N and WORK(IR) is LDA by N
							ldwrku = (*lda)
							ir = iu + ldwrku*(*n)
							ldwrkr = (*lda)
						} else if (*lwork) >= wrkbl+((*lda)+(*n))*(*n) {
							//                       WORK(IU) is LDA by N and WORK(IR) is N by N
							ldwrku = (*lda)
							ir = iu + ldwrku*(*n)
							ldwrkr = (*n)
						} else {
							//                       WORK(IU) is N by N and WORK(IR) is N by N
							ldwrku = (*n)
							ir = iu + ldwrku*(*n)
							ldwrkr = (*n)
						}
						itau = ir + ldwrkr*(*n)
						iwork = itau + (*n)

						//                    Compute A=Q*R, copying result to U
						//                    (CWorkspace: need 2*N*N+2*N, prefer 2*N*N+N+N*NB)
						//                    (RWorkspace: 0)
						Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('L', m, n, a, lda, u, ldu)

						//                    Generate Q in U
						//                    (CWorkspace: need 2*N*N+N+M, prefer 2*N*N+N+M*NB)
						//                    (RWorkspace: 0)
						Zungqr(m, m, n, u, ldu, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy R to WORK(IU), zeroing out below it
						Zlacpy('U', n, n, a, lda, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, work.CMatrixOff(iu, ldwrku, opts), &ldwrku)
						ie = 1
						itauq = itau
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Bidiagonalize R in WORK(IU), copying result to
						//                    WORK(IR)
						//                    (CWorkspace: need   2*N*N+3*N,
						//                                 prefer 2*N*N+2*N+2*N*NB)
						//                    (RWorkspace: need   N)
						Zgebrd(n, n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('U', n, n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr)

						//                    Generate left bidiagonalizing vectors in WORK(IU)
						//                    (CWorkspace: need 2*N*N+3*N, prefer 2*N*N+2*N+N*NB)
						//                    (RWorkspace: 0)
						Zungbr('Q', n, n, n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right bidiagonalizing vectors in WORK(IR)
						//                    (CWorkspace: need   2*N*N+3*N-1,
						//                                 prefer 2*N*N+2*N+(N-1)*NB)
						//                    (RWorkspace: 0)
						Zungbr('P', n, n, n, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IU) and computing
						//                    right singular vectors of R in WORK(IR)
						//                    (CWorkspace: need 2*N*N)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', n, n, n, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

						//                    Multiply Q in U by left singular vectors of R in
						//                    WORK(IU), storing result in A
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: 0)
						err = goblas.Zgemm(NoTrans, NoTrans, *m, *n, *n, cone, u, work.CMatrixOff(iu-1, ldwrku, opts), czero, a)

						//                    Copy left singular vectors of A from A to U
						Zlacpy('F', m, n, a, lda, u, ldu)

						//                    Copy right singular vectors of R from WORK(IR) to A
						Zlacpy('F', n, n, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, a, lda)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*n)

						//                    Compute A=Q*R, copying result to U
						//                    (CWorkspace: need 2*N, prefer N+N*NB)
						//                    (RWorkspace: 0)
						Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('L', m, n, a, lda, u, ldu)

						//                    Generate Q in U
						//                    (CWorkspace: need N+M, prefer N+M*NB)
						//                    (RWorkspace: 0)
						Zungqr(m, m, n, u, ldu, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = 1
						itauq = itau
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Zero out below R in A
						if (*n) > 1 {
							Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, a.Off(1, 0), lda)
						}

						//                    Bidiagonalize R in A
						//                    (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
						//                    (RWorkspace: need N)
						Zgebrd(n, n, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply Q in U by left bidiagonalizing vectors
						//                    in A
						//                    (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
						//                    (RWorkspace: 0)
						Zunmbr('Q', 'R', 'N', m, n, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right bidiagonalizing vectors in A
						//                    (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
						//                    (RWorkspace: 0)
						Zungbr('P', n, n, n, a, lda, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in A
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', n, n, m, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), a, lda, u, ldu, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

					}

				} else if wntvas {
					//                 Path 9 (M much larger than N, JOBU='A', JOBVT='S'
					//                         or 'A')
					//                 M left singular vectors to be computed in U and
					//                 N right singular vectors to be computed in VT
					if (*lwork) >= (*n)*(*n)+max((*n)+(*m), 3*(*n)) {
						//                    Sufficient workspace for a fast algorithm
						iu = 1
						if (*lwork) >= wrkbl+(*lda)*(*n) {
							//                       WORK(IU) is LDA by N
							ldwrku = (*lda)
						} else {
							//                       WORK(IU) is N by N
							ldwrku = (*n)
						}
						itau = iu + ldwrku*(*n)
						iwork = itau + (*n)

						//                    Compute A=Q*R, copying result to U
						//                    (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
						//                    (RWorkspace: 0)
						Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('L', m, n, a, lda, u, ldu)

						//                    Generate Q in U
						//                    (CWorkspace: need N*N+N+M, prefer N*N+N+M*NB)
						//                    (RWorkspace: 0)
						Zungqr(m, m, n, u, ldu, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy R to WORK(IU), zeroing out below it
						Zlacpy('U', n, n, a, lda, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, work.CMatrixOff(iu, ldwrku, opts), &ldwrku)
						ie = 1
						itauq = itau
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Bidiagonalize R in WORK(IU), copying result to VT
						//                    (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
						//                    (RWorkspace: need N)
						Zgebrd(n, n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('U', n, n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, vt, ldvt)

						//                    Generate left bidiagonalizing vectors in WORK(IU)
						//                    (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
						//                    (RWorkspace: 0)
						Zungbr('Q', n, n, n, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right bidiagonalizing vectors in VT
						//                    (CWorkspace: need   N*N+3*N-1,
						//                                 prefer N*N+2*N+(N-1)*NB)
						//                    (RWorkspace: need   0)
						Zungbr('P', n, n, n, vt, ldvt, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IU) and computing
						//                    right singular vectors of R in VT
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', n, n, n, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), vt, ldvt, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

						//                    Multiply Q in U by left singular vectors of R in
						//                    WORK(IU), storing result in A
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: 0)
						err = goblas.Zgemm(NoTrans, NoTrans, *m, *n, *n, cone, u, work.CMatrixOff(iu-1, ldwrku, opts), czero, a)

						//                    Copy left singular vectors of A from A to U
						Zlacpy('F', m, n, a, lda, u, ldu)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*n)

						//                    Compute A=Q*R, copying result to U
						//                    (CWorkspace: need 2*N, prefer N+N*NB)
						//                    (RWorkspace: 0)
						Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('L', m, n, a, lda, u, ldu)

						//                    Generate Q in U
						//                    (CWorkspace: need N+M, prefer N+M*NB)
						//                    (RWorkspace: 0)
						Zungqr(m, m, n, u, ldu, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy R from A to VT, zeroing out below it
						Zlacpy('U', n, n, a, lda, vt, ldvt)
						if (*n) > 1 {
							Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, vt.Off(1, 0), ldvt)
						}
						ie = 1
						itauq = itau
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Bidiagonalize R in VT
						//                    (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
						//                    (RWorkspace: need N)
						Zgebrd(n, n, vt, ldvt, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply Q in U by left bidiagonalizing vectors
						//                    in VT
						//                    (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
						//                    (RWorkspace: 0)
						Zunmbr('Q', 'R', 'N', m, n, n, vt, ldvt, work.Off(itauq-1), u, ldu, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right bidiagonalizing vectors in VT
						//                    (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
						//                    (RWorkspace: 0)
						Zungbr('P', n, n, n, vt, ldvt, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in VT
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', n, n, m, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), vt, ldvt, u, ldu, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

					}

				}

			}

		} else {
			//           M .LT. MNTHR
			//
			//           Path 10 (M at least N, but not much larger)
			//           Reduce to bidiagonal form without QR decomposition
			ie = 1
			itauq = 1
			itaup = itauq + (*n)
			iwork = itaup + (*n)

			//           Bidiagonalize A
			//           (CWorkspace: need 2*N+M, prefer 2*N+(M+N)*NB)
			//           (RWorkspace: need N)
			Zgebrd(m, n, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			if wntuas {
				//              If left singular vectors desired in U, copy result to U
				//              and generate left bidiagonalizing vectors in U
				//              (CWorkspace: need 2*N+NCU, prefer 2*N+NCU*NB)
				//              (RWorkspace: 0)
				Zlacpy('L', m, n, a, lda, u, ldu)
				if wntus {
					ncu = (*n)
				}
				if wntua {
					ncu = (*m)
				}
				Zungbr('Q', m, &ncu, n, u, ldu, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			}
			if wntvas {
				//              If right singular vectors desired in VT, copy result to
				//              VT and generate right bidiagonalizing vectors in VT
				//              (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
				//              (RWorkspace: 0)
				Zlacpy('U', n, n, a, lda, vt, ldvt)
				Zungbr('P', n, n, n, vt, ldvt, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			}
			if wntuo {
				//              If left singular vectors desired in A, generate left
				//              bidiagonalizing vectors in A
				//              (CWorkspace: need 3*N, prefer 2*N+N*NB)
				//              (RWorkspace: 0)
				Zungbr('Q', m, n, n, a, lda, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			}
			if wntvo {
				//              If right singular vectors desired in A, generate right
				//              bidiagonalizing vectors in A
				//              (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
				//              (RWorkspace: 0)
				Zungbr('P', n, n, n, a, lda, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			}
			irwork = ie + (*n)
			if wntuas || wntuo {
				nru = (*m)
			}
			if wntun {
				nru = 0
			}
			if wntvas || wntvo {
				ncvt = (*n)
			}
			if wntvn {
				ncvt = 0
			}
			if (!wntuo) && (!wntvo) {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in U and computing right singular
				//              vectors in VT
				//              (CWorkspace: 0)
				//              (RWorkspace: need BDSPAC)
				Zbdsqr('U', n, &ncvt, &nru, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), vt, ldvt, u, ldu, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)
			} else if (!wntuo) && wntvo {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in U and computing right singular
				//              vectors in A
				//              (CWorkspace: 0)
				//              (RWorkspace: need BDSPAC)
				Zbdsqr('U', n, &ncvt, &nru, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), a, lda, u, ldu, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)
			} else {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in A and computing right singular
				//              vectors in VT
				//              (CWorkspace: 0)
				//              (RWorkspace: need BDSPAC)
				Zbdsqr('U', n, &ncvt, &nru, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), vt, ldvt, a, lda, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)
			}

		}

	} else {
		//        A has more columns than rows. If A has sufficiently more
		//        columns than rows, first reduce using the LQ decomposition (if
		//        sufficient workspace available)
		if (*n) >= mnthr {

			if wntvn {
				//              Path 1t(N much larger than M, JOBVT='N')
				//              No right singular vectors to be computed
				itau = 1
				iwork = itau + (*m)

				//              Compute A=L*Q
				//              (CWorkspace: need 2*M, prefer M+M*NB)
				//              (RWorkspace: 0)
				Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

				//              Zero out above L
				Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, a.Off(0, 1), lda)
				ie = 1
				itauq = 1
				itaup = itauq + (*m)
				iwork = itaup + (*m)

				//              Bidiagonalize L in A
				//              (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
				//              (RWorkspace: need M)
				Zgebrd(m, m, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
				if wntuo || wntuas {
					//                 If left singular vectors desired, generate Q
					//                 (CWorkspace: need 3*M, prefer 2*M+M*NB)
					//                 (RWorkspace: 0)
					Zungbr('Q', m, m, m, a, lda, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
				}
				irwork = ie + (*m)
				nru = 0
				if wntuo || wntuas {
					nru = (*m)
				}

				//              Perform bidiagonal QR iteration, computing left singular
				//              vectors of A in A if desired
				//              (CWorkspace: 0)
				//              (RWorkspace: need BDSPAC)
				Zbdsqr('U', m, func() *int { y := 0; return &y }(), &nru, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), a, lda, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

				//              If left singular vectors desired in U, copy them there
				if wntuas {
					Zlacpy('F', m, m, a, lda, u, ldu)
				}

			} else if wntvo && wntun {
				//              Path 2t(N much larger than M, JOBU='N', JOBVT='O')
				//              M right singular vectors to be overwritten on A and
				//              no left singular vectors to be computed
				if (*lwork) >= (*m)*(*m)+3*(*m) {
					//                 Sufficient workspace for a fast algorithm
					ir = 1
					if (*lwork) >= max(wrkbl, (*lda)*(*n))+(*lda)*(*m) {
						//                    WORK(IU) is LDA by N and WORK(IR) is LDA by M
						ldwrku = (*lda)
						chunk = (*n)
						ldwrkr = (*lda)
					} else if (*lwork) >= max(wrkbl, (*lda)*(*n))+(*m)*(*m) {
						//                    WORK(IU) is LDA by N and WORK(IR) is M by M
						ldwrku = (*lda)
						chunk = (*n)
						ldwrkr = (*m)
					} else {
						//                    WORK(IU) is M by CHUNK and WORK(IR) is M by M
						ldwrku = (*m)
						chunk = ((*lwork) - (*m)*(*m)) / (*m)
						ldwrkr = (*m)
					}
					itau = ir + ldwrkr*(*m)
					iwork = itau + (*m)

					//                 Compute A=L*Q
					//                 (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
					//                 (RWorkspace: 0)
					Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Copy L to WORK(IR) and zero out above it
					Zlacpy('L', m, m, a, lda, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr)
					Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, work.CMatrixOff(ir+ldwrkr-1, ldwrkr, opts), &ldwrkr)

					//                 Generate Q in A
					//                 (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
					//                 (RWorkspace: 0)
					Zunglq(m, n, m, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					ie = 1
					itauq = itau
					itaup = itauq + (*m)
					iwork = itaup + (*m)

					//                 Bidiagonalize L in WORK(IR)
					//                 (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
					//                 (RWorkspace: need M)
					Zgebrd(m, m, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Generate right vectors bidiagonalizing L
					//                 (CWorkspace: need M*M+3*M-1, prefer M*M+2*M+(M-1)*NB)
					//                 (RWorkspace: 0)
					Zungbr('P', m, m, m, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					irwork = ie + (*m)

					//                 Perform bidiagonal QR iteration, computing right
					//                 singular vectors of L in WORK(IR)
					//                 (CWorkspace: need M*M)
					//                 (RWorkspace: need BDSPAC)
					Zbdsqr('U', m, m, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)
					iu = itauq

					//                 Multiply right singular vectors of L in WORK(IR) by Q
					//                 in A, storing result in WORK(IU) and copying to A
					//                 (CWorkspace: need M*M+M, prefer M*M+M*N)
					//                 (RWorkspace: 0)
					for i = 1; i <= (*n); i += chunk {
						blk = min((*n)-i+1, chunk)
						err = goblas.Zgemm(NoTrans, NoTrans, *m, blk, *m, cone, work.CMatrixOff(ir-1, ldwrkr, opts), a.Off(0, i-1), czero, work.CMatrixOff(iu-1, ldwrku, opts))
						Zlacpy('F', m, &blk, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, a.Off(0, i-1), lda)
					}

				} else {

					//                 Insufficient workspace for a fast algorithm
					ie = 1
					itauq = 1
					itaup = itauq + (*m)
					iwork = itaup + (*m)

					//                 Bidiagonalize A
					//                 (CWorkspace: need 2*M+N, prefer 2*M+(M+N)*NB)
					//                 (RWorkspace: need M)
					Zgebrd(m, n, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Generate right vectors bidiagonalizing A
					//                 (CWorkspace: need 3*M, prefer 2*M+M*NB)
					//                 (RWorkspace: 0)
					Zungbr('P', m, n, m, a, lda, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					irwork = ie + (*m)

					//                 Perform bidiagonal QR iteration, computing right
					//                 singular vectors of A in A
					//                 (CWorkspace: 0)
					//                 (RWorkspace: need BDSPAC)
					Zbdsqr('L', m, n, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), a, lda, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

				}

			} else if wntvo && wntuas {
				//              Path 3t(N much larger than M, JOBU='S' or 'A', JOBVT='O')
				//              M right singular vectors to be overwritten on A and
				//              M left singular vectors to be computed in U
				if (*lwork) >= (*m)*(*m)+3*(*m) {
					//                 Sufficient workspace for a fast algorithm
					ir = 1
					if (*lwork) >= max(wrkbl, (*lda)*(*n))+(*lda)*(*m) {
						//                    WORK(IU) is LDA by N and WORK(IR) is LDA by M
						ldwrku = (*lda)
						chunk = (*n)
						ldwrkr = (*lda)
					} else if (*lwork) >= max(wrkbl, (*lda)*(*n))+(*m)*(*m) {
						//                    WORK(IU) is LDA by N and WORK(IR) is M by M
						ldwrku = (*lda)
						chunk = (*n)
						ldwrkr = (*m)
					} else {
						//                    WORK(IU) is M by CHUNK and WORK(IR) is M by M
						ldwrku = (*m)
						chunk = ((*lwork) - (*m)*(*m)) / (*m)
						ldwrkr = (*m)
					}
					itau = ir + ldwrkr*(*m)
					iwork = itau + (*m)

					//                 Compute A=L*Q
					//                 (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
					//                 (RWorkspace: 0)
					Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Copy L to U, zeroing about above it
					Zlacpy('L', m, m, a, lda, u, ldu)
					Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, u.Off(0, 1), ldu)

					//                 Generate Q in A
					//                 (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
					//                 (RWorkspace: 0)
					Zunglq(m, n, m, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					ie = 1
					itauq = itau
					itaup = itauq + (*m)
					iwork = itaup + (*m)

					//                 Bidiagonalize L in U, copying result to WORK(IR)
					//                 (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
					//                 (RWorkspace: need M)
					Zgebrd(m, m, u, ldu, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					Zlacpy('U', m, m, u, ldu, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr)

					//                 Generate right vectors bidiagonalizing L in WORK(IR)
					//                 (CWorkspace: need M*M+3*M-1, prefer M*M+2*M+(M-1)*NB)
					//                 (RWorkspace: 0)
					Zungbr('P', m, m, m, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Generate left vectors bidiagonalizing L in U
					//                 (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB)
					//                 (RWorkspace: 0)
					Zungbr('Q', m, m, m, u, ldu, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					irwork = ie + (*m)

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of L in U, and computing right
					//                 singular vectors of L in WORK(IR)
					//                 (CWorkspace: need M*M)
					//                 (RWorkspace: need BDSPAC)
					Zbdsqr('U', m, m, m, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, u, ldu, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)
					iu = itauq

					//                 Multiply right singular vectors of L in WORK(IR) by Q
					//                 in A, storing result in WORK(IU) and copying to A
					//                 (CWorkspace: need M*M+M, prefer M*M+M*N))
					//                 (RWorkspace: 0)
					for i = 1; i <= (*n); i += chunk {
						blk = min((*n)-i+1, chunk)
						err = goblas.Zgemm(NoTrans, NoTrans, *m, blk, *m, cone, work.CMatrixOff(ir-1, ldwrkr, opts), a.Off(0, i-1), czero, work.CMatrixOff(iu-1, ldwrku, opts))
						Zlacpy('F', m, &blk, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, a.Off(0, i-1), lda)
					}

				} else {

					//                 Insufficient workspace for a fast algorithm
					itau = 1
					iwork = itau + (*m)

					//                 Compute A=L*Q
					//                 (CWorkspace: need 2*M, prefer M+M*NB)
					//                 (RWorkspace: 0)
					Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Copy L to U, zeroing out above it
					Zlacpy('L', m, m, a, lda, u, ldu)
					Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, u.Off(0, 1), ldu)

					//                 Generate Q in A
					//                 (CWorkspace: need 2*M, prefer M+M*NB)
					//                 (RWorkspace: 0)
					Zunglq(m, n, m, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					ie = 1
					itauq = itau
					itaup = itauq + (*m)
					iwork = itaup + (*m)

					//                 Bidiagonalize L in U
					//                 (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
					//                 (RWorkspace: need M)
					Zgebrd(m, m, u, ldu, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Multiply right vectors bidiagonalizing L by Q in A
					//                 (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
					//                 (RWorkspace: 0)
					Zunmbr('P', 'L', 'C', m, n, m, u, ldu, work.Off(itaup-1), a, lda, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Generate left vectors bidiagonalizing L in U
					//                 (CWorkspace: need 3*M, prefer 2*M+M*NB)
					//                 (RWorkspace: 0)
					Zungbr('Q', m, m, m, u, ldu, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					irwork = ie + (*m)

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of A in U and computing right
					//                 singular vectors of A in A
					//                 (CWorkspace: 0)
					//                 (RWorkspace: need BDSPAC)
					Zbdsqr('U', m, n, m, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), a, lda, u, ldu, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

				}

			} else if wntvs {

				if wntun {
					//                 Path 4t(N much larger than M, JOBU='N', JOBVT='S')
					//                 M right singular vectors to be computed in VT and
					//                 no left singular vectors to be computed
					if (*lwork) >= (*m)*(*m)+3*(*m) {
						//                    Sufficient workspace for a fast algorithm
						ir = 1
						if (*lwork) >= wrkbl+(*lda)*(*m) {
							//                       WORK(IR) is LDA by M
							ldwrkr = (*lda)
						} else {
							//                       WORK(IR) is M by M
							ldwrkr = (*m)
						}
						itau = ir + ldwrkr*(*m)
						iwork = itau + (*m)

						//                    Compute A=L*Q
						//                    (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
						//                    (RWorkspace: 0)
						Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy L to WORK(IR), zeroing out above it
						Zlacpy('L', m, m, a, lda, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr)
						Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, work.CMatrixOff(ir+ldwrkr-1, ldwrkr, opts), &ldwrkr)

						//                    Generate Q in A
						//                    (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
						//                    (RWorkspace: 0)
						Zunglq(m, n, m, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = 1
						itauq = itau
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Bidiagonalize L in WORK(IR)
						//                    (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
						//                    (RWorkspace: need M)
						Zgebrd(m, m, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right vectors bidiagonalizing L in
						//                    WORK(IR)
						//                    (CWorkspace: need M*M+3*M, prefer M*M+2*M+(M-1)*NB)
						//                    (RWorkspace: 0)
						Zungbr('P', m, m, m, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing right
						//                    singular vectors of L in WORK(IR)
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', m, m, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

						//                    Multiply right singular vectors of L in WORK(IR) by
						//                    Q in A, storing result in VT
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: 0)
						err = goblas.Zgemm(NoTrans, NoTrans, *m, *n, *m, cone, work.CMatrixOff(ir-1, ldwrkr, opts), a, czero, vt)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*m)

						//                    Compute A=L*Q
						//                    (CWorkspace: need 2*M, prefer M+M*NB)
						//                    (RWorkspace: 0)
						Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy result to VT
						Zlacpy('U', m, n, a, lda, vt, ldvt)

						//                    Generate Q in VT
						//                    (CWorkspace: need 2*M, prefer M+M*NB)
						//                    (RWorkspace: 0)
						Zunglq(m, n, m, vt, ldvt, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = 1
						itauq = itau
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Zero out above L in A
						Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, a.Off(0, 1), lda)

						//                    Bidiagonalize L in A
						//                    (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
						//                    (RWorkspace: need M)
						Zgebrd(m, m, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply right vectors bidiagonalizing L by Q in VT
						//                    (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
						//                    (RWorkspace: 0)
						Zunmbr('P', 'L', 'C', m, n, m, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing right
						//                    singular vectors of A in VT
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', m, n, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), vt, ldvt, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

					}

				} else if wntuo {
					//                 Path 5t(N much larger than M, JOBU='O', JOBVT='S')
					//                 M right singular vectors to be computed in VT and
					//                 M left singular vectors to be overwritten on A
					if (*lwork) >= 2*(*m)*(*m)+3*(*m) {
						//                    Sufficient workspace for a fast algorithm
						iu = 1
						if (*lwork) >= wrkbl+2*(*lda)*(*m) {
							//                       WORK(IU) is LDA by M and WORK(IR) is LDA by M
							ldwrku = (*lda)
							ir = iu + ldwrku*(*m)
							ldwrkr = (*lda)
						} else if (*lwork) >= wrkbl+((*lda)+(*m))*(*m) {
							//                       WORK(IU) is LDA by M and WORK(IR) is M by M
							ldwrku = (*lda)
							ir = iu + ldwrku*(*m)
							ldwrkr = (*m)
						} else {
							//                       WORK(IU) is M by M and WORK(IR) is M by M
							ldwrku = (*m)
							ir = iu + ldwrku*(*m)
							ldwrkr = (*m)
						}
						itau = ir + ldwrkr*(*m)
						iwork = itau + (*m)

						//                    Compute A=L*Q
						//                    (CWorkspace: need 2*M*M+2*M, prefer 2*M*M+M+M*NB)
						//                    (RWorkspace: 0)
						Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy L to WORK(IU), zeroing out below it
						Zlacpy('L', m, m, a, lda, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, work.CMatrixOff(iu+ldwrku-1, ldwrku, opts), &ldwrku)

						//                    Generate Q in A
						//                    (CWorkspace: need 2*M*M+2*M, prefer 2*M*M+M+M*NB)
						//                    (RWorkspace: 0)
						Zunglq(m, n, m, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = 1
						itauq = itau
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Bidiagonalize L in WORK(IU), copying result to
						//                    WORK(IR)
						//                    (CWorkspace: need   2*M*M+3*M,
						//                                 prefer 2*M*M+2*M+2*M*NB)
						//                    (RWorkspace: need   M)
						Zgebrd(m, m, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('L', m, m, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr)

						//                    Generate right bidiagonalizing vectors in WORK(IU)
						//                    (CWorkspace: need   2*M*M+3*M-1,
						//                                 prefer 2*M*M+2*M+(M-1)*NB)
						//                    (RWorkspace: 0)
						Zungbr('P', m, m, m, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left bidiagonalizing vectors in WORK(IR)
						//                    (CWorkspace: need 2*M*M+3*M, prefer 2*M*M+2*M+M*NB)
						//                    (RWorkspace: 0)
						Zungbr('Q', m, m, m, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of L in WORK(IR) and computing
						//                    right singular vectors of L in WORK(IU)
						//                    (CWorkspace: need 2*M*M)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', m, m, m, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

						//                    Multiply right singular vectors of L in WORK(IU) by
						//                    Q in A, storing result in VT
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: 0)
						err = goblas.Zgemm(NoTrans, NoTrans, *m, *n, *m, cone, work.CMatrixOff(iu-1, ldwrku, opts), a, czero, vt)

						//                    Copy left singular vectors of L to A
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: 0)
						Zlacpy('F', m, m, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, a, lda)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*m)

						//                    Compute A=L*Q, copying result to VT
						//                    (CWorkspace: need 2*M, prefer M+M*NB)
						//                    (RWorkspace: 0)
						Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('U', m, n, a, lda, vt, ldvt)

						//                    Generate Q in VT
						//                    (CWorkspace: need 2*M, prefer M+M*NB)
						//                    (RWorkspace: 0)
						Zunglq(m, n, m, vt, ldvt, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = 1
						itauq = itau
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Zero out above L in A
						Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, a.Off(0, 1), lda)

						//                    Bidiagonalize L in A
						//                    (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
						//                    (RWorkspace: need M)
						Zgebrd(m, m, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply right vectors bidiagonalizing L by Q in VT
						//                    (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
						//                    (RWorkspace: 0)
						Zunmbr('P', 'L', 'C', m, n, m, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left bidiagonalizing vectors of L in A
						//                    (CWorkspace: need 3*M, prefer 2*M+M*NB)
						//                    (RWorkspace: 0)
						Zungbr('Q', m, m, m, a, lda, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in A and computing right
						//                    singular vectors of A in VT
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', m, n, m, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), vt, ldvt, a, lda, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

					}

				} else if wntuas {
					//                 Path 6t(N much larger than M, JOBU='S' or 'A',
					//                         JOBVT='S')
					//                 M right singular vectors to be computed in VT and
					//                 M left singular vectors to be computed in U
					if (*lwork) >= (*m)*(*m)+3*(*m) {
						//                    Sufficient workspace for a fast algorithm
						iu = 1
						if (*lwork) >= wrkbl+(*lda)*(*m) {
							//                       WORK(IU) is LDA by N
							ldwrku = (*lda)
						} else {
							//                       WORK(IU) is LDA by M
							ldwrku = (*m)
						}
						itau = iu + ldwrku*(*m)
						iwork = itau + (*m)

						//                    Compute A=L*Q
						//                    (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
						//                    (RWorkspace: 0)
						Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy L to WORK(IU), zeroing out above it
						Zlacpy('L', m, m, a, lda, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, work.CMatrixOff(iu+ldwrku-1, ldwrku, opts), &ldwrku)

						//                    Generate Q in A
						//                    (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
						//                    (RWorkspace: 0)
						Zunglq(m, n, m, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = 1
						itauq = itau
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Bidiagonalize L in WORK(IU), copying result to U
						//                    (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
						//                    (RWorkspace: need M)
						Zgebrd(m, m, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('L', m, m, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, u, ldu)

						//                    Generate right bidiagonalizing vectors in WORK(IU)
						//                    (CWorkspace: need   M*M+3*M-1,
						//                                 prefer M*M+2*M+(M-1)*NB)
						//                    (RWorkspace: 0)
						Zungbr('P', m, m, m, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left bidiagonalizing vectors in U
						//                    (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB)
						//                    (RWorkspace: 0)
						Zungbr('Q', m, m, m, u, ldu, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of L in U and computing right
						//                    singular vectors of L in WORK(IU)
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', m, m, m, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, u, ldu, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

						//                    Multiply right singular vectors of L in WORK(IU) by
						//                    Q in A, storing result in VT
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: 0)
						err = goblas.Zgemm(NoTrans, NoTrans, *m, *n, *m, cone, work.CMatrixOff(iu-1, ldwrku, opts), a, czero, vt)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*m)

						//                    Compute A=L*Q, copying result to VT
						//                    (CWorkspace: need 2*M, prefer M+M*NB)
						//                    (RWorkspace: 0)
						Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('U', m, n, a, lda, vt, ldvt)

						//                    Generate Q in VT
						//                    (CWorkspace: need 2*M, prefer M+M*NB)
						//                    (RWorkspace: 0)
						Zunglq(m, n, m, vt, ldvt, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy L to U, zeroing out above it
						Zlacpy('L', m, m, a, lda, u, ldu)
						Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, u.Off(0, 1), ldu)
						ie = 1
						itauq = itau
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Bidiagonalize L in U
						//                    (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
						//                    (RWorkspace: need M)
						Zgebrd(m, m, u, ldu, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply right bidiagonalizing vectors in U by Q
						//                    in VT
						//                    (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
						//                    (RWorkspace: 0)
						Zunmbr('P', 'L', 'C', m, n, m, u, ldu, work.Off(itaup-1), vt, ldvt, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left bidiagonalizing vectors in U
						//                    (CWorkspace: need 3*M, prefer 2*M+M*NB)
						//                    (RWorkspace: 0)
						Zungbr('Q', m, m, m, u, ldu, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in VT
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', m, n, m, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), vt, ldvt, u, ldu, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

					}

				}

			} else if wntva {

				if wntun {
					//                 Path 7t(N much larger than M, JOBU='N', JOBVT='A')
					//                 N right singular vectors to be computed in VT and
					//                 no left singular vectors to be computed
					if (*lwork) >= (*m)*(*m)+max((*n)+(*m), 3*(*m)) {
						//                    Sufficient workspace for a fast algorithm
						ir = 1
						if (*lwork) >= wrkbl+(*lda)*(*m) {
							//                       WORK(IR) is LDA by M
							ldwrkr = (*lda)
						} else {
							//                       WORK(IR) is M by M
							ldwrkr = (*m)
						}
						itau = ir + ldwrkr*(*m)
						iwork = itau + (*m)

						//                    Compute A=L*Q, copying result to VT
						//                    (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
						//                    (RWorkspace: 0)
						Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('U', m, n, a, lda, vt, ldvt)

						//                    Copy L to WORK(IR), zeroing out above it
						Zlacpy('L', m, m, a, lda, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr)
						Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, work.CMatrixOff(ir+ldwrkr-1, ldwrkr, opts), &ldwrkr)

						//                    Generate Q in VT
						//                    (CWorkspace: need M*M+M+N, prefer M*M+M+N*NB)
						//                    (RWorkspace: 0)
						Zunglq(n, n, m, vt, ldvt, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = 1
						itauq = itau
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Bidiagonalize L in WORK(IR)
						//                    (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
						//                    (RWorkspace: need M)
						Zgebrd(m, m, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right bidiagonalizing vectors in WORK(IR)
						//                    (CWorkspace: need   M*M+3*M-1,
						//                                 prefer M*M+2*M+(M-1)*NB)
						//                    (RWorkspace: 0)
						Zungbr('P', m, m, m, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing right
						//                    singular vectors of L in WORK(IR)
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', m, m, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

						//                    Multiply right singular vectors of L in WORK(IR) by
						//                    Q in VT, storing result in A
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: 0)
						err = goblas.Zgemm(NoTrans, NoTrans, *m, *n, *m, cone, work.CMatrixOff(ir-1, ldwrkr, opts), vt, czero, a)

						//                    Copy right singular vectors of A from A to VT
						Zlacpy('F', m, n, a, lda, vt, ldvt)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*m)

						//                    Compute A=L*Q, copying result to VT
						//                    (CWorkspace: need 2*M, prefer M+M*NB)
						//                    (RWorkspace: 0)
						Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('U', m, n, a, lda, vt, ldvt)

						//                    Generate Q in VT
						//                    (CWorkspace: need M+N, prefer M+N*NB)
						//                    (RWorkspace: 0)
						Zunglq(n, n, m, vt, ldvt, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = 1
						itauq = itau
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Zero out above L in A
						Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, a.Off(0, 1), lda)

						//                    Bidiagonalize L in A
						//                    (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
						//                    (RWorkspace: need M)
						Zgebrd(m, m, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply right bidiagonalizing vectors in A by Q
						//                    in VT
						//                    (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
						//                    (RWorkspace: 0)
						Zunmbr('P', 'L', 'C', m, n, m, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing right
						//                    singular vectors of A in VT
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', m, n, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), vt, ldvt, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

					}

				} else if wntuo {
					//                 Path 8t(N much larger than M, JOBU='O', JOBVT='A')
					//                 N right singular vectors to be computed in VT and
					//                 M left singular vectors to be overwritten on A
					if (*lwork) >= 2*(*m)*(*m)+max((*n)+(*m), 3*(*m)) {
						//                    Sufficient workspace for a fast algorithm
						iu = 1
						if (*lwork) >= wrkbl+2*(*lda)*(*m) {
							//                       WORK(IU) is LDA by M and WORK(IR) is LDA by M
							ldwrku = (*lda)
							ir = iu + ldwrku*(*m)
							ldwrkr = (*lda)
						} else if (*lwork) >= wrkbl+((*lda)+(*m))*(*m) {
							//                       WORK(IU) is LDA by M and WORK(IR) is M by M
							ldwrku = (*lda)
							ir = iu + ldwrku*(*m)
							ldwrkr = (*m)
						} else {
							//                       WORK(IU) is M by M and WORK(IR) is M by M
							ldwrku = (*m)
							ir = iu + ldwrku*(*m)
							ldwrkr = (*m)
						}
						itau = ir + ldwrkr*(*m)
						iwork = itau + (*m)

						//                    Compute A=L*Q, copying result to VT
						//                    (CWorkspace: need 2*M*M+2*M, prefer 2*M*M+M+M*NB)
						//                    (RWorkspace: 0)
						Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('U', m, n, a, lda, vt, ldvt)

						//                    Generate Q in VT
						//                    (CWorkspace: need 2*M*M+M+N, prefer 2*M*M+M+N*NB)
						//                    (RWorkspace: 0)
						Zunglq(n, n, m, vt, ldvt, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy L to WORK(IU), zeroing out above it
						Zlacpy('L', m, m, a, lda, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, work.CMatrixOff(iu+ldwrku-1, ldwrku, opts), &ldwrku)
						ie = 1
						itauq = itau
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Bidiagonalize L in WORK(IU), copying result to
						//                    WORK(IR)
						//                    (CWorkspace: need   2*M*M+3*M,
						//                                 prefer 2*M*M+2*M+2*M*NB)
						//                    (RWorkspace: need   M)
						Zgebrd(m, m, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('L', m, m, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr)

						//                    Generate right bidiagonalizing vectors in WORK(IU)
						//                    (CWorkspace: need   2*M*M+3*M-1,
						//                                 prefer 2*M*M+2*M+(M-1)*NB)
						//                    (RWorkspace: 0)
						Zungbr('P', m, m, m, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left bidiagonalizing vectors in WORK(IR)
						//                    (CWorkspace: need 2*M*M+3*M, prefer 2*M*M+2*M+M*NB)
						//                    (RWorkspace: 0)
						Zungbr('Q', m, m, m, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of L in WORK(IR) and computing
						//                    right singular vectors of L in WORK(IU)
						//                    (CWorkspace: need 2*M*M)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', m, m, m, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

						//                    Multiply right singular vectors of L in WORK(IU) by
						//                    Q in VT, storing result in A
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: 0)
						err = goblas.Zgemm(NoTrans, NoTrans, *m, *n, *m, cone, work.CMatrixOff(iu-1, ldwrku, opts), vt, czero, a)

						//                    Copy right singular vectors of A from A to VT
						Zlacpy('F', m, n, a, lda, vt, ldvt)

						//                    Copy left singular vectors of A from WORK(IR) to A
						Zlacpy('F', m, m, work.CMatrixOff(ir-1, ldwrkr, opts), &ldwrkr, a, lda)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*m)

						//                    Compute A=L*Q, copying result to VT
						//                    (CWorkspace: need 2*M, prefer M+M*NB)
						//                    (RWorkspace: 0)
						Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('U', m, n, a, lda, vt, ldvt)

						//                    Generate Q in VT
						//                    (CWorkspace: need M+N, prefer M+N*NB)
						//                    (RWorkspace: 0)
						Zunglq(n, n, m, vt, ldvt, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = 1
						itauq = itau
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Zero out above L in A
						Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, a.Off(0, 1), lda)

						//                    Bidiagonalize L in A
						//                    (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
						//                    (RWorkspace: need M)
						Zgebrd(m, m, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply right bidiagonalizing vectors in A by Q
						//                    in VT
						//                    (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
						//                    (RWorkspace: 0)
						Zunmbr('P', 'L', 'C', m, n, m, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left bidiagonalizing vectors in A
						//                    (CWorkspace: need 3*M, prefer 2*M+M*NB)
						//                    (RWorkspace: 0)
						Zungbr('Q', m, m, m, a, lda, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in A and computing right
						//                    singular vectors of A in VT
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', m, n, m, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), vt, ldvt, a, lda, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

					}

				} else if wntuas {
					//                 Path 9t(N much larger than M, JOBU='S' or 'A',
					//                         JOBVT='A')
					//                 N right singular vectors to be computed in VT and
					//                 M left singular vectors to be computed in U
					if (*lwork) >= (*m)*(*m)+max((*n)+(*m), 3*(*m)) {
						//                    Sufficient workspace for a fast algorithm
						iu = 1
						if (*lwork) >= wrkbl+(*lda)*(*m) {
							//                       WORK(IU) is LDA by M
							ldwrku = (*lda)
						} else {
							//                       WORK(IU) is M by M
							ldwrku = (*m)
						}
						itau = iu + ldwrku*(*m)
						iwork = itau + (*m)

						//                    Compute A=L*Q, copying result to VT
						//                    (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
						//                    (RWorkspace: 0)
						Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('U', m, n, a, lda, vt, ldvt)

						//                    Generate Q in VT
						//                    (CWorkspace: need M*M+M+N, prefer M*M+M+N*NB)
						//                    (RWorkspace: 0)
						Zunglq(n, n, m, vt, ldvt, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy L to WORK(IU), zeroing out above it
						Zlacpy('L', m, m, a, lda, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, work.CMatrixOff(iu+ldwrku-1, ldwrku, opts), &ldwrku)
						ie = 1
						itauq = itau
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Bidiagonalize L in WORK(IU), copying result to U
						//                    (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
						//                    (RWorkspace: need M)
						Zgebrd(m, m, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('L', m, m, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, u, ldu)

						//                    Generate right bidiagonalizing vectors in WORK(IU)
						//                    (CWorkspace: need M*M+3*M, prefer M*M+2*M+(M-1)*NB)
						//                    (RWorkspace: 0)
						Zungbr('P', m, m, m, work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left bidiagonalizing vectors in U
						//                    (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB)
						//                    (RWorkspace: 0)
						Zungbr('Q', m, m, m, u, ldu, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of L in U and computing right
						//                    singular vectors of L in WORK(IU)
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', m, m, m, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), work.CMatrixOff(iu-1, ldwrku, opts), &ldwrku, u, ldu, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

						//                    Multiply right singular vectors of L in WORK(IU) by
						//                    Q in VT, storing result in A
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: 0)
						err = goblas.Zgemm(NoTrans, NoTrans, *m, *n, *m, cone, work.CMatrixOff(iu-1, ldwrku, opts), vt, czero, a)

						//                    Copy right singular vectors of A from A to VT
						Zlacpy('F', m, n, a, lda, vt, ldvt)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*m)

						//                    Compute A=L*Q, copying result to VT
						//                    (CWorkspace: need 2*M, prefer M+M*NB)
						//                    (RWorkspace: 0)
						Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Zlacpy('U', m, n, a, lda, vt, ldvt)

						//                    Generate Q in VT
						//                    (CWorkspace: need M+N, prefer M+N*NB)
						//                    (RWorkspace: 0)
						Zunglq(n, n, m, vt, ldvt, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy L to U, zeroing out above it
						Zlacpy('L', m, m, a, lda, u, ldu)
						Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, u.Off(0, 1), ldu)
						ie = 1
						itauq = itau
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Bidiagonalize L in U
						//                    (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
						//                    (RWorkspace: need M)
						Zgebrd(m, m, u, ldu, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply right bidiagonalizing vectors in U by Q
						//                    in VT
						//                    (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
						//                    (RWorkspace: 0)
						Zunmbr('P', 'L', 'C', m, n, m, u, ldu, work.Off(itaup-1), vt, ldvt, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left bidiagonalizing vectors in U
						//                    (CWorkspace: need 3*M, prefer 2*M+M*NB)
						//                    (RWorkspace: 0)
						Zungbr('Q', m, m, m, u, ldu, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						irwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in VT
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						Zbdsqr('U', m, n, m, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), vt, ldvt, u, ldu, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)

					}

				}

			}

		} else {
			//           N .LT. MNTHR
			//
			//           Path 10t(N greater than M, but not much larger)
			//           Reduce to bidiagonal form without LQ decomposition
			ie = 1
			itauq = 1
			itaup = itauq + (*m)
			iwork = itaup + (*m)

			//           Bidiagonalize A
			//           (CWorkspace: need 2*M+N, prefer 2*M+(M+N)*NB)
			//           (RWorkspace: M)
			Zgebrd(m, n, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			if wntuas {
				//              If left singular vectors desired in U, copy result to U
				//              and generate left bidiagonalizing vectors in U
				//              (CWorkspace: need 3*M-1, prefer 2*M+(M-1)*NB)
				//              (RWorkspace: 0)
				Zlacpy('L', m, m, a, lda, u, ldu)
				Zungbr('Q', m, m, n, u, ldu, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			}
			if wntvas {
				//              If right singular vectors desired in VT, copy result to
				//              VT and generate right bidiagonalizing vectors in VT
				//              (CWorkspace: need 2*M+NRVT, prefer 2*M+NRVT*NB)
				//              (RWorkspace: 0)
				Zlacpy('U', m, n, a, lda, vt, ldvt)
				if wntva {
					nrvt = (*n)
				}
				if wntvs {
					nrvt = (*m)
				}
				Zungbr('P', &nrvt, n, m, vt, ldvt, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			}
			if wntuo {
				//              If left singular vectors desired in A, generate left
				//              bidiagonalizing vectors in A
				//              (CWorkspace: need 3*M-1, prefer 2*M+(M-1)*NB)
				//              (RWorkspace: 0)
				Zungbr('Q', m, m, n, a, lda, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			}
			if wntvo {
				//              If right singular vectors desired in A, generate right
				//              bidiagonalizing vectors in A
				//              (CWorkspace: need 3*M, prefer 2*M+M*NB)
				//              (RWorkspace: 0)
				Zungbr('P', m, n, m, a, lda, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			}
			irwork = ie + (*m)
			if wntuas || wntuo {
				nru = (*m)
			}
			if wntun {
				nru = 0
			}
			if wntvas || wntvo {
				ncvt = (*n)
			}
			if wntvn {
				ncvt = 0
			}
			if (!wntuo) && (!wntvo) {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in U and computing right singular
				//              vectors in VT
				//              (CWorkspace: 0)
				//              (RWorkspace: need BDSPAC)
				Zbdsqr('L', m, &ncvt, &nru, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), vt, ldvt, u, ldu, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)
			} else if (!wntuo) && wntvo {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in U and computing right singular
				//              vectors in A
				//              (CWorkspace: 0)
				//              (RWorkspace: need BDSPAC)
				Zbdsqr('L', m, &ncvt, &nru, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), a, lda, u, ldu, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)
			} else {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in A and computing right singular
				//              vectors in VT
				//              (CWorkspace: 0)
				//              (RWorkspace: need BDSPAC)
				Zbdsqr('L', m, &ncvt, &nru, func() *int { y := 0; return &y }(), s, rwork.Off(ie-1), vt, ldvt, a, lda, cdum.CMatrix(1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwork-1), info)
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
