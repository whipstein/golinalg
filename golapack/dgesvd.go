package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgesvd computes the singular value decomposition (SVD) of a real
// M-by-N matrix A, optionally computing the left and/or right singular
// vectors. The SVD is written
//
//      A = U * SIGMA * transpose(V)
//
// where SIGMA is an M-by-N matrix which is zero except for its
// min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
// V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
// are the singular values of A; they are real and non-negative, and
// are returned in descending order.  The first min(m,n) columns of
// U and V are the left and right singular vectors of A.
//
// Note that the routine returns V**T, not V.
func Dgesvd(jobu, jobvt byte, m, n *int, a *mat.Matrix, lda *int, s *mat.Vector, u *mat.Matrix, ldu *int, vt *mat.Matrix, ldvt *int, work *mat.Vector, lwork *int, info *int) {
	var lquery, wntua, wntuas, wntun, wntuo, wntus, wntva, wntvas, wntvn, wntvo, wntvs bool
	var anrm, bignum, eps, one, smlnum, zero float64
	var bdspac, blk, chunk, i, ie, ierr, ir, iscl, itau, itaup, itauq, iu, iwork, ldwrkr, ldwrku, lworkDgebrd, lworkDgelqf, lworkDgeqrf, lworkDorgbrP, lworkDorgbrQ, lworkDorglqM, lworkDorglqN, lworkDorgqrM, lworkDorgqrN, maxwrk, minmn, minwrk, mnthr, ncu, ncvt, nru, nrvt, wrkbl int

	dum := vf(1)

	zero = 0.0
	one = 1.0

	//     Test the input arguments
	(*info) = 0
	minmn = minint(*m, *n)
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

	if !(wntua || wntus || wntuo || wntun) {
		(*info) = -1
	} else if !(wntva || wntvs || wntvo || wntvn) || (wntvo && wntuo) {
		(*info) = -2
	} else if (*m) < 0 {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*lda) < maxint(1, *m) {
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
	//       NB refers to the optimal block size for the immediately
	//       following subroutine, as returned by ILAENV.)
	if (*info) == 0 {
		minwrk = 1
		maxwrk = 1
		if (*m) >= (*n) && minmn > 0 {
			//           Compute space needed for DBDSQR
			mnthr = Ilaenv(func() *int { y := 6; return &y }(), []byte("DGESVD"), []byte{jobu, jobvt}, m, n, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }())
			bdspac = 5 * (*n)
			//           Compute space needed for DGEQRF
			Dgeqrf(m, n, a, lda, dum, dum, toPtr(-1), &ierr)
			lworkDgeqrf = int(dum.Get(0))
			//           Compute space needed for DORGQR
			Dorgqr(m, n, n, a, lda, dum, dum, toPtr(-1), &ierr)
			lworkDorgqrN = int(dum.Get(0))
			Dorgqr(m, m, n, a, lda, dum, dum, toPtr(-1), &ierr)
			lworkDorgqrM = int(dum.Get(0))
			//           Compute space needed for DGEBRD
			Dgebrd(n, n, a, lda, s, dum, dum, dum, dum, toPtr(-1), &ierr)
			lworkDgebrd = int(dum.Get(0))
			//           Compute space needed for DORGBR P
			Dorgbr('P', n, n, n, a, lda, dum, dum, toPtr(-1), &ierr)
			lworkDorgbrP = int(dum.Get(0))
			//           Compute space needed for DORGBR Q
			Dorgbr('Q', n, n, n, a, lda, dum, dum, toPtr(-1), &ierr)
			lworkDorgbrQ = int(dum.Get(0))
			//
			if (*m) >= mnthr {
				if wntun {
					//                 Path 1 (M much larger than N, JOBU='N')
					maxwrk = (*n) + lworkDgeqrf
					maxwrk = maxint(maxwrk, 3*(*n)+lworkDgebrd)
					if wntvo || wntvas {
						maxwrk = maxint(maxwrk, 3*(*n)+lworkDorgbrP)
					}
					maxwrk = maxint(maxwrk, bdspac)
					minwrk = maxint(4*(*n), bdspac)
				} else if wntuo && wntvn {
					//                 Path 2 (M much larger than N, JOBU='O', JOBVT='N')
					wrkbl = (*n) + lworkDgeqrf
					wrkbl = maxint(wrkbl, (*n)+lworkDorgqrN)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDgebrd)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDorgbrQ)
					wrkbl = maxint(wrkbl, bdspac)
					maxwrk = maxint((*n)*(*n)+wrkbl, (*n)*(*n)+(*m)*(*n)+(*n))
					minwrk = maxint(3*(*n)+(*m), bdspac)
				} else if wntuo && wntvas {
					//                 Path 3 (M much larger than N, JOBU='O', JOBVT='S' or
					//                 'A')
					wrkbl = (*n) + lworkDgeqrf
					wrkbl = maxint(wrkbl, (*n)+lworkDorgqrN)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDgebrd)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDorgbrQ)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDorgbrP)
					wrkbl = maxint(wrkbl, bdspac)
					maxwrk = maxint((*n)*(*n)+wrkbl, (*n)*(*n)+(*m)*(*n)+(*n))
					minwrk = maxint(3*(*n)+(*m), bdspac)
				} else if wntus && wntvn {
					//                 Path 4 (M much larger than N, JOBU='S', JOBVT='N')
					wrkbl = (*n) + lworkDgeqrf
					wrkbl = maxint(wrkbl, (*n)+lworkDorgqrN)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDgebrd)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDorgbrQ)
					wrkbl = maxint(wrkbl, bdspac)
					maxwrk = (*n)*(*n) + wrkbl
					minwrk = maxint(3*(*n)+(*m), bdspac)
				} else if wntus && wntvo {
					//                 Path 5 (M much larger than N, JOBU='S', JOBVT='O')
					wrkbl = (*n) + lworkDgeqrf
					wrkbl = maxint(wrkbl, (*n)+lworkDorgqrN)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDgebrd)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDorgbrQ)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDorgbrP)
					wrkbl = maxint(wrkbl, bdspac)
					maxwrk = 2*(*n)*(*n) + wrkbl
					minwrk = maxint(3*(*n)+(*m), bdspac)
				} else if wntus && wntvas {
					//                 Path 6 (M much larger than N, JOBU='S', JOBVT='S' or
					//                 'A')
					wrkbl = (*n) + lworkDgeqrf
					wrkbl = maxint(wrkbl, (*n)+lworkDorgqrN)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDgebrd)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDorgbrQ)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDorgbrP)
					wrkbl = maxint(wrkbl, bdspac)
					maxwrk = (*n)*(*n) + wrkbl
					minwrk = maxint(3*(*n)+(*m), bdspac)
				} else if wntua && wntvn {
					//                 Path 7 (M much larger than N, JOBU='A', JOBVT='N')
					wrkbl = (*n) + lworkDgeqrf
					wrkbl = maxint(wrkbl, (*n)+lworkDorgqrM)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDgebrd)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDorgbrQ)
					wrkbl = maxint(wrkbl, bdspac)
					maxwrk = (*n)*(*n) + wrkbl
					minwrk = maxint(3*(*n)+(*m), bdspac)
				} else if wntua && wntvo {
					//                 Path 8 (M much larger than N, JOBU='A', JOBVT='O')
					wrkbl = (*n) + lworkDgeqrf
					wrkbl = maxint(wrkbl, (*n)+lworkDorgqrM)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDgebrd)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDorgbrQ)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDorgbrP)
					wrkbl = maxint(wrkbl, bdspac)
					maxwrk = 2*(*n)*(*n) + wrkbl
					minwrk = maxint(3*(*n)+(*m), bdspac)
				} else if wntua && wntvas {
					//                 Path 9 (M much larger than N, JOBU='A', JOBVT='S' or
					//                 'A')
					wrkbl = (*n) + lworkDgeqrf
					wrkbl = maxint(wrkbl, (*n)+lworkDorgqrM)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDgebrd)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDorgbrQ)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDorgbrP)
					wrkbl = maxint(wrkbl, bdspac)
					maxwrk = (*n)*(*n) + wrkbl
					minwrk = maxint(3*(*n)+(*m), bdspac)
				}
			} else {
				//              Path 10 (M at least N, but not much larger)
				Dgebrd(m, n, a, lda, s, dum, dum, dum, dum, toPtr(-1), &ierr)
				lworkDgebrd = int(dum.Get(0))
				maxwrk = 3*(*n) + lworkDgebrd
				if wntus || wntuo {
					Dorgbr('Q', m, n, n, a, lda, dum, dum, toPtr(-1), &ierr)
					lworkDorgbrQ = int(dum.Get(0))
					maxwrk = maxint(maxwrk, 3*(*n)+lworkDorgbrQ)
				}
				if wntua {
					Dorgbr('Q', m, m, n, a, lda, dum, dum, toPtr(-1), &ierr)
					lworkDorgbrQ = int(dum.Get(0))
					maxwrk = maxint(maxwrk, 3*(*n)+lworkDorgbrQ)
				}
				if !wntvn {
					maxwrk = maxint(maxwrk, 3*(*n)+lworkDorgbrP)
				}
				maxwrk = maxint(maxwrk, bdspac)
				minwrk = maxint(3*(*n)+(*m), bdspac)
			}
		} else if minmn > 0 {
			//           Compute space needed for DBDSQR
			mnthr = Ilaenv(func() *int { y := 6; return &y }(), []byte("DGESVD"), []byte{jobu, jobvt}, m, n, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }())
			bdspac = 5 * (*m)
			//           Compute space needed for DGELQF
			Dgelqf(m, n, a, lda, dum, dum, toPtr(-1), &ierr)
			lworkDgelqf = int(dum.Get(0))
			//           Compute space needed for DORGLQ
			Dorglq(n, n, m, dum.Matrix(*n, opts), n, dum, dum, toPtr(-1), &ierr)
			lworkDorglqN = int(dum.Get(0))
			Dorglq(m, n, m, a, lda, dum, dum, toPtr(-1), &ierr)
			lworkDorglqM = int(dum.Get(0))
			//           Compute space needed for DGEBRD
			Dgebrd(m, m, a, lda, s, dum, dum, dum, dum, toPtr(-1), &ierr)
			lworkDgebrd = int(dum.Get(0))
			//            Compute space needed for DORGBR P
			Dorgbr('P', m, m, m, a, n, dum, dum, toPtr(-1), &ierr)
			lworkDorgbrP = int(dum.Get(0))
			//           Compute space needed for DORGBR Q
			Dorgbr('Q', m, m, m, a, n, dum, dum, toPtr(-1), &ierr)
			lworkDorgbrQ = int(dum.Get(0))
			if (*n) >= mnthr {
				if wntvn {
					//                 Path 1t(N much larger than M, JOBVT='N')
					maxwrk = (*m) + lworkDgelqf
					maxwrk = maxint(maxwrk, 3*(*m)+lworkDgebrd)
					if wntuo || wntuas {
						maxwrk = maxint(maxwrk, 3*(*m)+lworkDorgbrQ)
					}
					maxwrk = maxint(maxwrk, bdspac)
					minwrk = maxint(4*(*m), bdspac)
				} else if wntvo && wntun {
					//                 Path 2t(N much larger than M, JOBU='N', JOBVT='O')
					wrkbl = (*m) + lworkDgelqf
					wrkbl = maxint(wrkbl, (*m)+lworkDorglqM)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDgebrd)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDorgbrP)
					wrkbl = maxint(wrkbl, bdspac)
					maxwrk = maxint((*m)*(*m)+wrkbl, (*m)*(*m)+(*m)*(*n)+(*m))
					minwrk = maxint(3*(*m)+(*n), bdspac)
				} else if wntvo && wntuas {
					//                 Path 3t(N much larger than M, JOBU='S' or 'A',
					//                 JOBVT='O')
					wrkbl = (*m) + lworkDgelqf
					wrkbl = maxint(wrkbl, (*m)+lworkDorglqM)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDgebrd)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDorgbrP)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDorgbrQ)
					wrkbl = maxint(wrkbl, bdspac)
					maxwrk = maxint((*m)*(*m)+wrkbl, (*m)*(*m)+(*m)*(*n)+(*m))
					minwrk = maxint(3*(*m)+(*n), bdspac)
				} else if wntvs && wntun {
					//                 Path 4t(N much larger than M, JOBU='N', JOBVT='S')
					wrkbl = (*m) + lworkDgelqf
					wrkbl = maxint(wrkbl, (*m)+lworkDorglqM)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDgebrd)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDorgbrP)
					wrkbl = maxint(wrkbl, bdspac)
					maxwrk = (*m)*(*m) + wrkbl
					minwrk = maxint(3*(*m)+(*n), bdspac)
				} else if wntvs && wntuo {
					//                 Path 5t(N much larger than M, JOBU='O', JOBVT='S')
					wrkbl = (*m) + lworkDgelqf
					wrkbl = maxint(wrkbl, (*m)+lworkDorglqM)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDgebrd)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDorgbrP)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDorgbrQ)
					wrkbl = maxint(wrkbl, bdspac)
					maxwrk = 2*(*m)*(*m) + wrkbl
					minwrk = maxint(3*(*m)+(*n), bdspac)
				} else if wntvs && wntuas {
					//                 Path 6t(N much larger than M, JOBU='S' or 'A',
					//                 JOBVT='S')
					wrkbl = (*m) + lworkDgelqf
					wrkbl = maxint(wrkbl, (*m)+lworkDorglqM)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDgebrd)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDorgbrP)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDorgbrQ)
					wrkbl = maxint(wrkbl, bdspac)
					maxwrk = (*m)*(*m) + wrkbl
					minwrk = maxint(3*(*m)+(*n), bdspac)
				} else if wntva && wntun {
					//                 Path 7t(N much larger than M, JOBU='N', JOBVT='A')
					wrkbl = (*m) + lworkDgelqf
					wrkbl = maxint(wrkbl, (*m)+lworkDorglqN)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDgebrd)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDorgbrP)
					wrkbl = maxint(wrkbl, bdspac)
					maxwrk = (*m)*(*m) + wrkbl
					minwrk = maxint(3*(*m)+(*n), bdspac)
				} else if wntva && wntuo {
					//                 Path 8t(N much larger than M, JOBU='O', JOBVT='A')
					wrkbl = (*m) + lworkDgelqf
					wrkbl = maxint(wrkbl, (*m)+lworkDorglqN)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDgebrd)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDorgbrP)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDorgbrQ)
					wrkbl = maxint(wrkbl, bdspac)
					maxwrk = 2*(*m)*(*m) + wrkbl
					minwrk = maxint(3*(*m)+(*n), bdspac)
				} else if wntva && wntuas {
					//                 Path 9t(N much larger than M, JOBU='S' or 'A',
					//                 JOBVT='A')
					wrkbl = (*m) + lworkDgelqf
					wrkbl = maxint(wrkbl, (*m)+lworkDorglqN)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDgebrd)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDorgbrP)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDorgbrQ)
					wrkbl = maxint(wrkbl, bdspac)
					maxwrk = (*m)*(*m) + wrkbl
					minwrk = maxint(3*(*m)+(*n), bdspac)
				}
			} else {
				//              Path 10t(N greater than M, but not much larger)
				Dgebrd(m, n, a, lda, s, dum, dum, dum, dum, toPtr(-1), &ierr)
				lworkDgebrd = int(dum.Get(0))
				maxwrk = 3*(*m) + lworkDgebrd
				if wntvs || wntvo {
					//                Compute space needed for DORGBR P
					Dorgbr('P', m, n, m, a, n, dum, dum, toPtr(-1), &ierr)
					lworkDorgbrP = int(dum.Get(0))
					maxwrk = maxint(maxwrk, 3*(*m)+lworkDorgbrP)
				}
				if wntva {
					Dorgbr('P', n, n, m, a, n, dum, dum, toPtr(-1), &ierr)
					lworkDorgbrP = int(dum.Get(0))
					maxwrk = maxint(maxwrk, 3*(*m)+lworkDorgbrP)
				}
				if !wntun {
					maxwrk = maxint(maxwrk, 3*(*m)+lworkDorgbrQ)
				}
				maxwrk = maxint(maxwrk, bdspac)
				minwrk = maxint(3*(*m)+(*n), bdspac)
			}
		}
		maxwrk = maxint(maxwrk, minwrk)
		work.Set(0, float64(maxwrk))

		if (*lwork) < minwrk && !lquery {
			(*info) = -13
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGESVD"), -(*info))
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
	anrm = Dlange('M', m, n, a, lda, dum)
	iscl = 0
	if anrm > zero && anrm < smlnum {
		iscl = 1
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &smlnum, m, n, a, lda, &ierr)
	} else if anrm > bignum {
		iscl = 1
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &bignum, m, n, a, lda, &ierr)
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
				//              (Workspace: need 2*N, prefer N + N*NB)
				Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

				//              Zero out below R
				if (*n) > 1 {
					Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, a.Off(1, 0), lda)
				}
				ie = 1
				itauq = ie + (*n)
				itaup = itauq + (*n)
				iwork = itaup + (*n)

				//              Bidiagonalize R in A
				//              (Workspace: need 4*N, prefer 3*N + 2*N*NB)
				Dgebrd(n, n, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
				ncvt = 0
				if wntvo || wntvas {
					//                 If right singular vectors desired, generate P'.
					//                 (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
					Dorgbr('P', n, n, n, a, lda, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					ncvt = (*n)
				}
				iwork = ie + (*n)

				//              Perform bidiagonal QR iteration, computing right
				//              singular vectors of A in A if desired
				//              (Workspace: need BDSPAC)
				Dbdsqr('U', n, &ncvt, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), s, work.Off(ie-1), a, lda, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

				//              If right singular vectors desired in VT, copy them there
				if wntvas {
					Dlacpy('F', n, n, a, lda, vt, ldvt)
				}

			} else if wntuo && wntvn {
				//              Path 2 (M much larger than N, JOBU='O', JOBVT='N')
				//              N left singular vectors to be overwritten on A and
				//              no right singular vectors to be computed
				if (*lwork) >= (*n)*(*n)+maxint(4*(*n), bdspac) {
					//                 Sufficient workspace for a fast algorithm
					ir = 1
					if (*lwork) >= maxint(wrkbl, (*lda)*(*n)+(*n))+(*lda)*(*n) {
						//                    WORK(IU) is LDA by N, WORK(IR) is LDA by N
						ldwrku = (*lda)
						ldwrkr = (*lda)
					} else if (*lwork) >= maxint(wrkbl, (*lda)*(*n)+(*n))+(*n)*(*n) {
						//                    WORK(IU) is LDA by N, WORK(IR) is N by N
						ldwrku = (*lda)
						ldwrkr = (*n)
					} else {
						//                    WORK(IU) is LDWRKU by N, WORK(IR) is N by N
						ldwrku = ((*lwork) - (*n)*(*n) - (*n)) / (*n)
						ldwrkr = (*n)
					}
					itau = ir + ldwrkr*(*n)
					iwork = itau + (*n)

					//                 Compute A=Q*R
					//                 (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
					Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Copy R to WORK(IR) and zero out below it
					Dlacpy('U', n, n, a, lda, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr)
					Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, work.MatrixOff(ir+1-1, ldwrkr, opts), &ldwrkr)

					//                 Generate Q in A
					//                 (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
					Dorgqr(m, n, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					ie = itau
					itauq = ie + (*n)
					itaup = itauq + (*n)
					iwork = itaup + (*n)

					//                 Bidiagonalize R in WORK(IR)
					//                 (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
					Dgebrd(n, n, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Generate left vectors bidiagonalizing R
					//                 (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
					Dorgbr('Q', n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					iwork = ie + (*n)

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of R in WORK(IR)
					//                 (Workspace: need N*N + BDSPAC)
					Dbdsqr('U', n, func() *int { y := 0; return &y }(), n, func() *int { y := 0; return &y }(), s, work.Off(ie-1), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)
					iu = ie + (*n)

					//                 Multiply Q in A by left singular vectors of R in
					//                 WORK(IR), storing result in WORK(IU) and copying to A
					//                 (Workspace: need N*N + 2*N, prefer N*N + M*N + N)
					for i = 1; i <= (*m); i += ldwrku {
						chunk = minint((*m)-i+1, ldwrku)
						goblas.Dgemm(NoTrans, NoTrans, &chunk, n, n, &one, a.Off(i-1, 0), lda, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, &zero, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Dlacpy('F', &chunk, n, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, a.Off(i-1, 0), lda)
					}

				} else {
					//                 Insufficient workspace for a fast algorithm
					ie = 1
					itauq = ie + (*n)
					itaup = itauq + (*n)
					iwork = itaup + (*n)

					//                 Bidiagonalize A
					//                 (Workspace: need 3*N + M, prefer 3*N + (M + N)*NB)
					Dgebrd(m, n, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Generate left vectors bidiagonalizing A
					//                 (Workspace: need 4*N, prefer 3*N + N*NB)
					Dorgbr('Q', m, n, n, a, lda, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					iwork = ie + (*n)

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of A in A
					//                 (Workspace: need BDSPAC)
					Dbdsqr('U', n, func() *int { y := 0; return &y }(), m, func() *int { y := 0; return &y }(), s, work.Off(ie-1), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), a, lda, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

				}

			} else if wntuo && wntvas {
				//              Path 3 (M much larger than N, JOBU='O', JOBVT='S' or 'A')
				//              N left singular vectors to be overwritten on A and
				//              N right singular vectors to be computed in VT
				if (*lwork) >= (*n)*(*n)+maxint(4*(*n), bdspac) {
					//                 Sufficient workspace for a fast algorithm
					ir = 1
					if (*lwork) >= maxint(wrkbl, (*lda)*(*n)+(*n))+(*lda)*(*n) {
						//                    WORK(IU) is LDA by N and WORK(IR) is LDA by N
						ldwrku = (*lda)
						ldwrkr = (*lda)
					} else if (*lwork) >= maxint(wrkbl, (*lda)*(*n)+(*n))+(*n)*(*n) {
						//                    WORK(IU) is LDA by N and WORK(IR) is N by N
						ldwrku = (*lda)
						ldwrkr = (*n)
					} else {
						//                    WORK(IU) is LDWRKU by N and WORK(IR) is N by N
						ldwrku = ((*lwork) - (*n)*(*n) - (*n)) / (*n)
						ldwrkr = (*n)
					}
					itau = ir + ldwrkr*(*n)
					iwork = itau + (*n)

					//                 Compute A=Q*R
					//                 (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
					Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Copy R to VT, zeroing out below it
					Dlacpy('U', n, n, a, lda, vt, ldvt)
					if (*n) > 1 {
						Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, vt.Off(1, 0), ldvt)
					}

					//                 Generate Q in A
					//                 (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
					Dorgqr(m, n, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					ie = itau
					itauq = ie + (*n)
					itaup = itauq + (*n)
					iwork = itaup + (*n)

					//                 Bidiagonalize R in VT, copying result to WORK(IR)
					//                 (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
					Dgebrd(n, n, vt, ldvt, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					Dlacpy('L', n, n, vt, ldvt, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr)

					//                 Generate left vectors bidiagonalizing R in WORK(IR)
					//                 (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
					Dorgbr('Q', n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Generate right vectors bidiagonalizing R in VT
					//                 (Workspace: need N*N + 4*N-1, prefer N*N + 3*N + (N-1)*NB)
					Dorgbr('P', n, n, n, vt, ldvt, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					iwork = ie + (*n)

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of R in WORK(IR) and computing right
					//                 singular vectors of R in VT
					//                 (Workspace: need N*N + BDSPAC)
					Dbdsqr('U', n, n, n, func() *int { y := 0; return &y }(), s, work.Off(ie-1), vt, ldvt, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)
					iu = ie + (*n)

					//                 Multiply Q in A by left singular vectors of R in
					//                 WORK(IR), storing result in WORK(IU) and copying to A
					//                 (Workspace: need N*N + 2*N, prefer N*N + M*N + N)
					for i = 1; i <= (*m); i += ldwrku {
						chunk = minint((*m)-i+1, ldwrku)
						goblas.Dgemm(NoTrans, NoTrans, &chunk, n, n, &one, a.Off(i-1, 0), lda, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, &zero, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Dlacpy('F', &chunk, n, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, a.Off(i-1, 0), lda)
					}

				} else {
					//                 Insufficient workspace for a fast algorithm
					itau = 1
					iwork = itau + (*n)

					//                 Compute A=Q*R
					//                 (Workspace: need 2*N, prefer N + N*NB)
					Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Copy R to VT, zeroing out below it
					Dlacpy('U', n, n, a, lda, vt, ldvt)
					if (*n) > 1 {
						Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, vt.Off(1, 0), ldvt)
					}

					//                 Generate Q in A
					//                 (Workspace: need 2*N, prefer N + N*NB)
					Dorgqr(m, n, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					ie = itau
					itauq = ie + (*n)
					itaup = itauq + (*n)
					iwork = itaup + (*n)

					//                 Bidiagonalize R in VT
					//                 (Workspace: need 4*N, prefer 3*N + 2*N*NB)
					Dgebrd(n, n, vt, ldvt, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Multiply Q in A by left vectors bidiagonalizing R
					//                 (Workspace: need 3*N + M, prefer 3*N + M*NB)
					Dormbr('Q', 'R', 'N', m, n, n, vt, ldvt, work.Off(itauq-1), a, lda, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Generate right vectors bidiagonalizing R in VT
					//                 (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
					Dorgbr('P', n, n, n, vt, ldvt, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					iwork = ie + (*n)

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of A in A and computing right
					//                 singular vectors of A in VT
					//                 (Workspace: need BDSPAC)
					Dbdsqr('U', n, n, m, func() *int { y := 0; return &y }(), s, work.Off(ie-1), vt, ldvt, a, lda, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

				}

			} else if wntus {

				if wntvn {
					//                 Path 4 (M much larger than N, JOBU='S', JOBVT='N')
					//                 N left singular vectors to be computed in U and
					//                 no right singular vectors to be computed
					if (*lwork) >= (*n)*(*n)+maxint(4*(*n), bdspac) {
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
						//                    (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
						Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy R to WORK(IR), zeroing out below it
						Dlacpy('U', n, n, a, lda, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr)
						Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, work.MatrixOff(ir+1-1, ldwrkr, opts), &ldwrkr)

						//                    Generate Q in A
						//                    (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
						Dorgqr(m, n, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = itau
						itauq = ie + (*n)
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Bidiagonalize R in WORK(IR)
						//                    (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
						Dgebrd(n, n, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left vectors bidiagonalizing R in WORK(IR)
						//                    (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
						Dorgbr('Q', n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IR)
						//                    (Workspace: need N*N + BDSPAC)
						Dbdsqr('U', n, func() *int { y := 0; return &y }(), n, func() *int { y := 0; return &y }(), s, work.Off(ie-1), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

						//                    Multiply Q in A by left singular vectors of R in
						//                    WORK(IR), storing result in U
						//                    (Workspace: need N*N)
						goblas.Dgemm(NoTrans, NoTrans, m, n, n, &one, a, lda, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, &zero, u, ldu)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*n)

						//                    Compute A=Q*R, copying result to U
						//                    (Workspace: need 2*N, prefer N + N*NB)
						Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('L', m, n, a, lda, u, ldu)

						//                    Generate Q in U
						//                    (Workspace: need 2*N, prefer N + N*NB)
						Dorgqr(m, n, n, u, ldu, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = itau
						itauq = ie + (*n)
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Zero out below R in A
						if (*n) > 1 {
							Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, a.Off(1, 0), lda)
						}

						//                    Bidiagonalize R in A
						//                    (Workspace: need 4*N, prefer 3*N + 2*N*NB)
						Dgebrd(n, n, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply Q in U by left vectors bidiagonalizing R
						//                    (Workspace: need 3*N + M, prefer 3*N + M*NB)
						Dormbr('Q', 'R', 'N', m, n, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U
						//                    (Workspace: need BDSPAC)
						Dbdsqr('U', n, func() *int { y := 0; return &y }(), m, func() *int { y := 0; return &y }(), s, work.Off(ie-1), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), u, ldu, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

					}

				} else if wntvo {
					//                 Path 5 (M much larger than N, JOBU='S', JOBVT='O')
					//                 N left singular vectors to be computed in U and
					//                 N right singular vectors to be overwritten on A
					if (*lwork) >= 2*(*n)*(*n)+maxint(4*(*n), bdspac) {
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
						//                    (Workspace: need 2*N*N + 2*N, prefer 2*N*N + N + N*NB)
						Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy R to WORK(IU), zeroing out below it
						Dlacpy('U', n, n, a, lda, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, work.MatrixOff(iu+1-1, ldwrku, opts), &ldwrku)

						//                    Generate Q in A
						//                    (Workspace: need 2*N*N + 2*N, prefer 2*N*N + N + N*NB)
						Dorgqr(m, n, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = itau
						itauq = ie + (*n)
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Bidiagonalize R in WORK(IU), copying result to
						//                    WORK(IR)
						//                    (Workspace: need 2*N*N + 4*N,
						//                                prefer 2*N*N+3*N+2*N*NB)
						Dgebrd(n, n, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('U', n, n, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr)

						//                    Generate left bidiagonalizing vectors in WORK(IU)
						//                    (Workspace: need 2*N*N + 4*N, prefer 2*N*N + 3*N + N*NB)
						Dorgbr('Q', n, n, n, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right bidiagonalizing vectors in WORK(IR)
						//                    (Workspace: need 2*N*N + 4*N-1,
						//                                prefer 2*N*N+3*N+(N-1)*NB)
						Dorgbr('P', n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IU) and computing
						//                    right singular vectors of R in WORK(IR)
						//                    (Workspace: need 2*N*N + BDSPAC)
						Dbdsqr('U', n, n, n, func() *int { y := 0; return &y }(), s, work.Off(ie-1), work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

						//                    Multiply Q in A by left singular vectors of R in
						//                    WORK(IU), storing result in U
						//                    (Workspace: need N*N)
						goblas.Dgemm(NoTrans, NoTrans, m, n, n, &one, a, lda, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, &zero, u, ldu)

						//                    Copy right singular vectors of R to A
						//                    (Workspace: need N*N)
						Dlacpy('F', n, n, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, a, lda)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*n)

						//                    Compute A=Q*R, copying result to U
						//                    (Workspace: need 2*N, prefer N + N*NB)
						Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('L', m, n, a, lda, u, ldu)

						//                    Generate Q in U
						//                    (Workspace: need 2*N, prefer N + N*NB)
						Dorgqr(m, n, n, u, ldu, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = itau
						itauq = ie + (*n)
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Zero out below R in A
						if (*n) > 1 {
							Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, a.Off(1, 0), lda)
						}

						//                    Bidiagonalize R in A
						//                    (Workspace: need 4*N, prefer 3*N + 2*N*NB)
						Dgebrd(n, n, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply Q in U by left vectors bidiagonalizing R
						//                    (Workspace: need 3*N + M, prefer 3*N + M*NB)
						Dormbr('Q', 'R', 'N', m, n, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right vectors bidiagonalizing R in A
						//                    (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
						Dorgbr('P', n, n, n, a, lda, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in A
						//                    (Workspace: need BDSPAC)
						Dbdsqr('U', n, n, m, func() *int { y := 0; return &y }(), s, work.Off(ie-1), a, lda, u, ldu, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

					}

				} else if wntvas {
					//                 Path 6 (M much larger than N, JOBU='S', JOBVT='S'
					//                         or 'A')
					//                 N left singular vectors to be computed in U and
					//                 N right singular vectors to be computed in VT
					if (*lwork) >= (*n)*(*n)+maxint(4*(*n), bdspac) {
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
						//                    (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
						Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy R to WORK(IU), zeroing out below it
						Dlacpy('U', n, n, a, lda, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, work.MatrixOff(iu+1-1, ldwrku, opts), &ldwrku)

						//                    Generate Q in A
						//                    (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
						Dorgqr(m, n, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = itau
						itauq = ie + (*n)
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Bidiagonalize R in WORK(IU), copying result to VT
						//                    (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
						Dgebrd(n, n, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('U', n, n, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, vt, ldvt)

						//                    Generate left bidiagonalizing vectors in WORK(IU)
						//                    (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
						Dorgbr('Q', n, n, n, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right bidiagonalizing vectors in VT
						//                    (Workspace: need N*N + 4*N-1,
						//                                prefer N*N+3*N+(N-1)*NB)
						Dorgbr('P', n, n, n, vt, ldvt, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IU) and computing
						//                    right singular vectors of R in VT
						//                    (Workspace: need N*N + BDSPAC)
						Dbdsqr('U', n, n, n, func() *int { y := 0; return &y }(), s, work.Off(ie-1), vt, ldvt, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

						//                    Multiply Q in A by left singular vectors of R in
						//                    WORK(IU), storing result in U
						//                    (Workspace: need N*N)
						goblas.Dgemm(NoTrans, NoTrans, m, n, n, &one, a, lda, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, &zero, u, ldu)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*n)

						//                    Compute A=Q*R, copying result to U
						//                    (Workspace: need 2*N, prefer N + N*NB)
						Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('L', m, n, a, lda, u, ldu)

						//                    Generate Q in U
						//                    (Workspace: need 2*N, prefer N + N*NB)
						Dorgqr(m, n, n, u, ldu, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy R to VT, zeroing out below it
						Dlacpy('U', n, n, a, lda, vt, ldvt)
						if (*n) > 1 {
							Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, vt.Off(1, 0), ldvt)
						}
						ie = itau
						itauq = ie + (*n)
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Bidiagonalize R in VT
						//                    (Workspace: need 4*N, prefer 3*N + 2*N*NB)
						Dgebrd(n, n, vt, ldvt, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply Q in U by left bidiagonalizing vectors
						//                    in VT
						//                    (Workspace: need 3*N + M, prefer 3*N + M*NB)
						Dormbr('Q', 'R', 'N', m, n, n, vt, ldvt, work.Off(itauq-1), u, ldu, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right bidiagonalizing vectors in VT
						//                    (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
						Dorgbr('P', n, n, n, vt, ldvt, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in VT
						//                    (Workspace: need BDSPAC)
						Dbdsqr('U', n, n, m, func() *int { y := 0; return &y }(), s, work.Off(ie-1), vt, ldvt, u, ldu, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

					}

				}

			} else if wntua {

				if wntvn {
					//                 Path 7 (M much larger than N, JOBU='A', JOBVT='N')
					//                 M left singular vectors to be computed in U and
					//                 no right singular vectors to be computed
					if (*lwork) >= (*n)*(*n)+maxint((*n)+(*m), 4*(*n), bdspac) {
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
						//                    (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
						Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('L', m, n, a, lda, u, ldu)

						//                    Copy R to WORK(IR), zeroing out below it
						Dlacpy('U', n, n, a, lda, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr)
						Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, work.MatrixOff(ir+1-1, ldwrkr, opts), &ldwrkr)

						//                    Generate Q in U
						//                    (Workspace: need N*N + N + M, prefer N*N + N + M*NB)
						Dorgqr(m, m, n, u, ldu, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = itau
						itauq = ie + (*n)
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Bidiagonalize R in WORK(IR)
						//                    (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
						Dgebrd(n, n, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left bidiagonalizing vectors in WORK(IR)
						//                    (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
						Dorgbr('Q', n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IR)
						//                    (Workspace: need N*N + BDSPAC)
						Dbdsqr('U', n, func() *int { y := 0; return &y }(), n, func() *int { y := 0; return &y }(), s, work.Off(ie-1), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

						//                    Multiply Q in U by left singular vectors of R in
						//                    WORK(IR), storing result in A
						//                    (Workspace: need N*N)
						goblas.Dgemm(NoTrans, NoTrans, m, n, n, &one, u, ldu, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, &zero, a, lda)

						//                    Copy left singular vectors of A from A to U
						Dlacpy('F', m, n, a, lda, u, ldu)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*n)

						//                    Compute A=Q*R, copying result to U
						//                    (Workspace: need 2*N, prefer N + N*NB)
						Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('L', m, n, a, lda, u, ldu)

						//                    Generate Q in U
						//                    (Workspace: need N + M, prefer N + M*NB)
						Dorgqr(m, m, n, u, ldu, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = itau
						itauq = ie + (*n)
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Zero out below R in A
						if (*n) > 1 {
							Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, a.Off(1, 0), lda)
						}

						//                    Bidiagonalize R in A
						//                    (Workspace: need 4*N, prefer 3*N + 2*N*NB)
						Dgebrd(n, n, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply Q in U by left bidiagonalizing vectors
						//                    in A
						//                    (Workspace: need 3*N + M, prefer 3*N + M*NB)
						Dormbr('Q', 'R', 'N', m, n, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U
						//                    (Workspace: need BDSPAC)
						Dbdsqr('U', n, func() *int { y := 0; return &y }(), m, func() *int { y := 0; return &y }(), s, work.Off(ie-1), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), u, ldu, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

					}

				} else if wntvo {
					//                 Path 8 (M much larger than N, JOBU='A', JOBVT='O')
					//                 M left singular vectors to be computed in U and
					//                 N right singular vectors to be overwritten on A
					if (*lwork) >= 2*(*n)*(*n)+maxint((*n)+(*m), 4*(*n), bdspac) {
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
						//                    (Workspace: need 2*N*N + 2*N, prefer 2*N*N + N + N*NB)
						Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('L', m, n, a, lda, u, ldu)

						//                    Generate Q in U
						//                    (Workspace: need 2*N*N + N + M, prefer 2*N*N + N + M*NB)
						Dorgqr(m, m, n, u, ldu, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy R to WORK(IU), zeroing out below it
						Dlacpy('U', n, n, a, lda, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, work.MatrixOff(iu+1-1, ldwrku, opts), &ldwrku)
						ie = itau
						itauq = ie + (*n)
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Bidiagonalize R in WORK(IU), copying result to
						//                    WORK(IR)
						//                    (Workspace: need 2*N*N + 4*N,
						//                                prefer 2*N*N+3*N+2*N*NB)
						Dgebrd(n, n, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('U', n, n, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr)

						//                    Generate left bidiagonalizing vectors in WORK(IU)
						//                    (Workspace: need 2*N*N + 4*N, prefer 2*N*N + 3*N + N*NB)
						Dorgbr('Q', n, n, n, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right bidiagonalizing vectors in WORK(IR)
						//                    (Workspace: need 2*N*N + 4*N-1,
						//                                prefer 2*N*N+3*N+(N-1)*NB)
						Dorgbr('P', n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IU) and computing
						//                    right singular vectors of R in WORK(IR)
						//                    (Workspace: need 2*N*N + BDSPAC)
						Dbdsqr('U', n, n, n, func() *int { y := 0; return &y }(), s, work.Off(ie-1), work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

						//                    Multiply Q in U by left singular vectors of R in
						//                    WORK(IU), storing result in A
						//                    (Workspace: need N*N)
						goblas.Dgemm(NoTrans, NoTrans, m, n, n, &one, u, ldu, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, &zero, a, lda)

						//                    Copy left singular vectors of A from A to U
						Dlacpy('F', m, n, a, lda, u, ldu)

						//                    Copy right singular vectors of R from WORK(IR) to A
						Dlacpy('F', n, n, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, a, lda)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*n)

						//                    Compute A=Q*R, copying result to U
						//                    (Workspace: need 2*N, prefer N + N*NB)
						Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('L', m, n, a, lda, u, ldu)

						//                    Generate Q in U
						//                    (Workspace: need N + M, prefer N + M*NB)
						Dorgqr(m, m, n, u, ldu, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = itau
						itauq = ie + (*n)
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Zero out below R in A
						if (*n) > 1 {
							Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, a.Off(1, 0), lda)
						}

						//                    Bidiagonalize R in A
						//                    (Workspace: need 4*N, prefer 3*N + 2*N*NB)
						Dgebrd(n, n, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply Q in U by left bidiagonalizing vectors
						//                    in A
						//                    (Workspace: need 3*N + M, prefer 3*N + M*NB)
						Dormbr('Q', 'R', 'N', m, n, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right bidiagonalizing vectors in A
						//                    (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
						Dorgbr('P', n, n, n, a, lda, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in A
						//                    (Workspace: need BDSPAC)
						Dbdsqr('U', n, n, m, func() *int { y := 0; return &y }(), s, work.Off(ie-1), a, lda, u, ldu, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

					}

				} else if wntvas {
					//                 Path 9 (M much larger than N, JOBU='A', JOBVT='S'
					//                         or 'A')
					//                 M left singular vectors to be computed in U and
					//                 N right singular vectors to be computed in VT
					if (*lwork) >= (*n)*(*n)+maxint((*n)+(*m), 4*(*n), bdspac) {
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
						//                    (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
						Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('L', m, n, a, lda, u, ldu)

						//                    Generate Q in U
						//                    (Workspace: need N*N + N + M, prefer N*N + N + M*NB)
						Dorgqr(m, m, n, u, ldu, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy R to WORK(IU), zeroing out below it
						Dlacpy('U', n, n, a, lda, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, work.MatrixOff(iu+1-1, ldwrku, opts), &ldwrku)
						ie = itau
						itauq = ie + (*n)
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Bidiagonalize R in WORK(IU), copying result to VT
						//                    (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
						Dgebrd(n, n, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('U', n, n, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, vt, ldvt)

						//                    Generate left bidiagonalizing vectors in WORK(IU)
						//                    (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
						Dorgbr('Q', n, n, n, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right bidiagonalizing vectors in VT
						//                    (Workspace: need N*N + 4*N-1,
						//                                prefer N*N+3*N+(N-1)*NB)
						Dorgbr('P', n, n, n, vt, ldvt, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IU) and computing
						//                    right singular vectors of R in VT
						//                    (Workspace: need N*N + BDSPAC)
						Dbdsqr('U', n, n, n, func() *int { y := 0; return &y }(), s, work.Off(ie-1), vt, ldvt, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

						//                    Multiply Q in U by left singular vectors of R in
						//                    WORK(IU), storing result in A
						//                    (Workspace: need N*N)
						goblas.Dgemm(NoTrans, NoTrans, m, n, n, &one, u, ldu, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, &zero, a, lda)

						//                    Copy left singular vectors of A from A to U
						Dlacpy('F', m, n, a, lda, u, ldu)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*n)

						//                    Compute A=Q*R, copying result to U
						//                    (Workspace: need 2*N, prefer N + N*NB)
						Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('L', m, n, a, lda, u, ldu)

						//                    Generate Q in U
						//                    (Workspace: need N + M, prefer N + M*NB)
						Dorgqr(m, m, n, u, ldu, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy R from A to VT, zeroing out below it
						Dlacpy('U', n, n, a, lda, vt, ldvt)
						if (*n) > 1 {
							Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, vt.Off(1, 0), ldvt)
						}
						ie = itau
						itauq = ie + (*n)
						itaup = itauq + (*n)
						iwork = itaup + (*n)

						//                    Bidiagonalize R in VT
						//                    (Workspace: need 4*N, prefer 3*N + 2*N*NB)
						Dgebrd(n, n, vt, ldvt, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply Q in U by left bidiagonalizing vectors
						//                    in VT
						//                    (Workspace: need 3*N + M, prefer 3*N + M*NB)
						Dormbr('Q', 'R', 'N', m, n, n, vt, ldvt, work.Off(itauq-1), u, ldu, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right bidiagonalizing vectors in VT
						//                    (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
						Dorgbr('P', n, n, n, vt, ldvt, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*n)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in VT
						//                    (Workspace: need BDSPAC)
						Dbdsqr('U', n, n, m, func() *int { y := 0; return &y }(), s, work.Off(ie-1), vt, ldvt, u, ldu, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

					}

				}

			}

		} else {
			//           M .LT. MNTHR
			//
			//           Path 10 (M at least N, but not much larger)
			//           Reduce to bidiagonal form without QR decomposition
			ie = 1
			itauq = ie + (*n)
			itaup = itauq + (*n)
			iwork = itaup + (*n)

			//           Bidiagonalize A
			//           (Workspace: need 3*N + M, prefer 3*N + (M + N)*NB)
			Dgebrd(m, n, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			if wntuas {
				//              If left singular vectors desired in U, copy result to U
				//              and generate left bidiagonalizing vectors in U
				//              (Workspace: need 3*N + NCU, prefer 3*N + NCU*NB)
				Dlacpy('L', m, n, a, lda, u, ldu)
				if wntus {
					ncu = (*n)
				}
				if wntua {
					ncu = (*m)
				}
				Dorgbr('Q', m, &ncu, n, u, ldu, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			}
			if wntvas {
				//              If right singular vectors desired in VT, copy result to
				//              VT and generate right bidiagonalizing vectors in VT
				//              (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
				Dlacpy('U', n, n, a, lda, vt, ldvt)
				Dorgbr('P', n, n, n, vt, ldvt, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			}
			if wntuo {
				//              If left singular vectors desired in A, generate left
				//              bidiagonalizing vectors in A
				//              (Workspace: need 4*N, prefer 3*N + N*NB)
				Dorgbr('Q', m, n, n, a, lda, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			}
			if wntvo {
				//              If right singular vectors desired in A, generate right
				//              bidiagonalizing vectors in A
				//              (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
				Dorgbr('P', n, n, n, a, lda, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			}
			iwork = ie + (*n)
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
				//              (Workspace: need BDSPAC)
				Dbdsqr('U', n, &ncvt, &nru, func() *int { y := 0; return &y }(), s, work.Off(ie-1), vt, ldvt, u, ldu, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)
			} else if (!wntuo) && wntvo {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in U and computing right singular
				//              vectors in A
				//              (Workspace: need BDSPAC)
				Dbdsqr('U', n, &ncvt, &nru, func() *int { y := 0; return &y }(), s, work.Off(ie-1), a, lda, u, ldu, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)
			} else {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in A and computing right singular
				//              vectors in VT
				//              (Workspace: need BDSPAC)
				Dbdsqr('U', n, &ncvt, &nru, func() *int { y := 0; return &y }(), s, work.Off(ie-1), vt, ldvt, a, lda, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)
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
				//              (Workspace: need 2*M, prefer M + M*NB)
				Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

				//              Zero out above L
				Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, a.Off(0, 1), lda)
				ie = 1
				itauq = ie + (*m)
				itaup = itauq + (*m)
				iwork = itaup + (*m)

				//              Bidiagonalize L in A
				//              (Workspace: need 4*M, prefer 3*M + 2*M*NB)
				Dgebrd(m, m, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
				if wntuo || wntuas {
					//                 If left singular vectors desired, generate Q
					//                 (Workspace: need 4*M, prefer 3*M + M*NB)
					Dorgbr('Q', m, m, m, a, lda, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
				}
				iwork = ie + (*m)
				nru = 0
				if wntuo || wntuas {
					nru = (*m)
				}

				//              Perform bidiagonal QR iteration, computing left singular
				//              vectors of A in A if desired
				//              (Workspace: need BDSPAC)
				Dbdsqr('U', m, func() *int { y := 0; return &y }(), &nru, func() *int { y := 0; return &y }(), s, work.Off(ie-1), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), a, lda, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

				//              If left singular vectors desired in U, copy them there
				if wntuas {
					Dlacpy('F', m, m, a, lda, u, ldu)
				}

			} else if wntvo && wntun {
				//              Path 2t(N much larger than M, JOBU='N', JOBVT='O')
				//              M right singular vectors to be overwritten on A and
				//              no left singular vectors to be computed
				if (*lwork) >= (*m)*(*m)+maxint(4*(*m), bdspac) {
					//                 Sufficient workspace for a fast algorithm
					ir = 1
					if (*lwork) >= maxint(wrkbl, (*lda)*(*n)+(*m))+(*lda)*(*m) {
						//                    WORK(IU) is LDA by N and WORK(IR) is LDA by M
						ldwrku = (*lda)
						chunk = (*n)
						ldwrkr = (*lda)
					} else if (*lwork) >= maxint(wrkbl, (*lda)*(*n)+(*m))+(*m)*(*m) {
						//                    WORK(IU) is LDA by N and WORK(IR) is M by M
						ldwrku = (*lda)
						chunk = (*n)
						ldwrkr = (*m)
					} else {
						//                    WORK(IU) is M by CHUNK and WORK(IR) is M by M
						ldwrku = (*m)
						chunk = ((*lwork) - (*m)*(*m) - (*m)) / (*m)
						ldwrkr = (*m)
					}
					itau = ir + ldwrkr*(*m)
					iwork = itau + (*m)

					//                 Compute A=L*Q
					//                 (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
					Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Copy L to WORK(IR) and zero out above it
					Dlacpy('L', m, m, a, lda, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr)
					Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, work.MatrixOff(ir+ldwrkr-1, ldwrkr, opts), &ldwrkr)

					//                 Generate Q in A
					//                 (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
					Dorglq(m, n, m, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					ie = itau
					itauq = ie + (*m)
					itaup = itauq + (*m)
					iwork = itaup + (*m)

					//                 Bidiagonalize L in WORK(IR)
					//                 (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
					Dgebrd(m, m, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Generate right vectors bidiagonalizing L
					//                 (Workspace: need M*M + 4*M-1, prefer M*M + 3*M + (M-1)*NB)
					Dorgbr('P', m, m, m, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					iwork = ie + (*m)

					//                 Perform bidiagonal QR iteration, computing right
					//                 singular vectors of L in WORK(IR)
					//                 (Workspace: need M*M + BDSPAC)
					Dbdsqr('U', m, m, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), s, work.Off(ie-1), work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)
					iu = ie + (*m)

					//                 Multiply right singular vectors of L in WORK(IR) by Q
					//                 in A, storing result in WORK(IU) and copying to A
					//                 (Workspace: need M*M + 2*M, prefer M*M + M*N + M)
					for i = 1; i <= (*n); i += chunk {
						blk = minint((*n)-i+1, chunk)
						goblas.Dgemm(NoTrans, NoTrans, m, &blk, m, &one, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, a.Off(0, i-1), lda, &zero, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Dlacpy('F', m, &blk, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, a.Off(0, i-1), lda)
					}

				} else {
					//                 Insufficient workspace for a fast algorithm
					ie = 1
					itauq = ie + (*m)
					itaup = itauq + (*m)
					iwork = itaup + (*m)

					//                 Bidiagonalize A
					//                 (Workspace: need 3*M + N, prefer 3*M + (M + N)*NB)
					Dgebrd(m, n, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Generate right vectors bidiagonalizing A
					//                 (Workspace: need 4*M, prefer 3*M + M*NB)
					Dorgbr('P', m, n, m, a, lda, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					iwork = ie + (*m)

					//                 Perform bidiagonal QR iteration, computing right
					//                 singular vectors of A in A
					//                 (Workspace: need BDSPAC)
					Dbdsqr('L', m, n, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), s, work.Off(ie-1), a, lda, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

				}

			} else if wntvo && wntuas {
				//              Path 3t(N much larger than M, JOBU='S' or 'A', JOBVT='O')
				//              M right singular vectors to be overwritten on A and
				//              M left singular vectors to be computed in U
				if (*lwork) >= (*m)*(*m)+maxint(4*(*m), bdspac) {
					//                 Sufficient workspace for a fast algorithm
					ir = 1
					if (*lwork) >= maxint(wrkbl, (*lda)*(*n)+(*m))+(*lda)*(*m) {
						//                    WORK(IU) is LDA by N and WORK(IR) is LDA by M
						ldwrku = (*lda)
						chunk = (*n)
						ldwrkr = (*lda)
					} else if (*lwork) >= maxint(wrkbl, (*lda)*(*n)+(*m))+(*m)*(*m) {
						//                    WORK(IU) is LDA by N and WORK(IR) is M by M
						ldwrku = (*lda)
						chunk = (*n)
						ldwrkr = (*m)
					} else {
						//                    WORK(IU) is M by CHUNK and WORK(IR) is M by M
						ldwrku = (*m)
						chunk = ((*lwork) - (*m)*(*m) - (*m)) / (*m)
						ldwrkr = (*m)
					}
					itau = ir + ldwrkr*(*m)
					iwork = itau + (*m)

					//                 Compute A=L*Q
					//                 (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
					Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Copy L to U, zeroing about above it
					Dlacpy('L', m, m, a, lda, u, ldu)
					Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, u.Off(0, 1), ldu)

					//                 Generate Q in A
					//                 (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
					Dorglq(m, n, m, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					ie = itau
					itauq = ie + (*m)
					itaup = itauq + (*m)
					iwork = itaup + (*m)

					//                 Bidiagonalize L in U, copying result to WORK(IR)
					//                 (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
					Dgebrd(m, m, u, ldu, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					Dlacpy('U', m, m, u, ldu, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr)

					//                 Generate right vectors bidiagonalizing L in WORK(IR)
					//                 (Workspace: need M*M + 4*M-1, prefer M*M + 3*M + (M-1)*NB)
					Dorgbr('P', m, m, m, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Generate left vectors bidiagonalizing L in U
					//                 (Workspace: need M*M + 4*M, prefer M*M + 3*M + M*NB)
					Dorgbr('Q', m, m, m, u, ldu, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					iwork = ie + (*m)

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of L in U, and computing right
					//                 singular vectors of L in WORK(IR)
					//                 (Workspace: need M*M + BDSPAC)
					Dbdsqr('U', m, m, m, func() *int { y := 0; return &y }(), s, work.Off(ie-1), work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, u, ldu, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)
					iu = ie + (*m)

					//                 Multiply right singular vectors of L in WORK(IR) by Q
					//                 in A, storing result in WORK(IU) and copying to A
					//                 (Workspace: need M*M + 2*M, prefer M*M + M*N + M))
					for i = 1; i <= (*n); i += chunk {
						blk = minint((*n)-i+1, chunk)
						goblas.Dgemm(NoTrans, NoTrans, m, &blk, m, &one, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, a.Off(0, i-1), lda, &zero, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Dlacpy('F', m, &blk, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, a.Off(0, i-1), lda)
					}

				} else {
					//                 Insufficient workspace for a fast algorithm
					itau = 1
					iwork = itau + (*m)

					//                 Compute A=L*Q
					//                 (Workspace: need 2*M, prefer M + M*NB)
					Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Copy L to U, zeroing out above it
					Dlacpy('L', m, m, a, lda, u, ldu)
					Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, u.Off(0, 1), ldu)

					//                 Generate Q in A
					//                 (Workspace: need 2*M, prefer M + M*NB)
					Dorglq(m, n, m, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					ie = itau
					itauq = ie + (*m)
					itaup = itauq + (*m)
					iwork = itaup + (*m)

					//                 Bidiagonalize L in U
					//                 (Workspace: need 4*M, prefer 3*M + 2*M*NB)
					Dgebrd(m, m, u, ldu, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Multiply right vectors bidiagonalizing L by Q in A
					//                 (Workspace: need 3*M + N, prefer 3*M + N*NB)
					Dormbr('P', 'L', 'T', m, n, m, u, ldu, work.Off(itaup-1), a, lda, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

					//                 Generate left vectors bidiagonalizing L in U
					//                 (Workspace: need 4*M, prefer 3*M + M*NB)
					Dorgbr('Q', m, m, m, u, ldu, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
					iwork = ie + (*m)

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of A in U and computing right
					//                 singular vectors of A in A
					//                 (Workspace: need BDSPAC)
					Dbdsqr('U', m, n, m, func() *int { y := 0; return &y }(), s, work.Off(ie-1), a, lda, u, ldu, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

				}

			} else if wntvs {

				if wntun {
					//                 Path 4t(N much larger than M, JOBU='N', JOBVT='S')
					//                 M right singular vectors to be computed in VT and
					//                 no left singular vectors to be computed
					if (*lwork) >= (*m)*(*m)+maxint(4*(*m), bdspac) {
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
						//                    (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
						Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy L to WORK(IR), zeroing out above it
						Dlacpy('L', m, m, a, lda, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr)
						Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, work.MatrixOff(ir+ldwrkr-1, ldwrkr, opts), &ldwrkr)

						//                    Generate Q in A
						//                    (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
						Dorglq(m, n, m, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = itau
						itauq = ie + (*m)
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Bidiagonalize L in WORK(IR)
						//                    (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
						Dgebrd(m, m, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right vectors bidiagonalizing L in
						//                    WORK(IR)
						//                    (Workspace: need M*M + 4*M, prefer M*M + 3*M + (M-1)*NB)
						Dorgbr('P', m, m, m, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing right
						//                    singular vectors of L in WORK(IR)
						//                    (Workspace: need M*M + BDSPAC)
						Dbdsqr('U', m, m, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), s, work.Off(ie-1), work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

						//                    Multiply right singular vectors of L in WORK(IR) by
						//                    Q in A, storing result in VT
						//                    (Workspace: need M*M)
						goblas.Dgemm(NoTrans, NoTrans, m, n, m, &one, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, a, lda, &zero, vt, ldvt)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*m)

						//                    Compute A=L*Q
						//                    (Workspace: need 2*M, prefer M + M*NB)
						Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy result to VT
						Dlacpy('U', m, n, a, lda, vt, ldvt)

						//                    Generate Q in VT
						//                    (Workspace: need 2*M, prefer M + M*NB)
						Dorglq(m, n, m, vt, ldvt, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = itau
						itauq = ie + (*m)
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Zero out above L in A
						Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, a.Off(0, 1), lda)

						//                    Bidiagonalize L in A
						//                    (Workspace: need 4*M, prefer 3*M + 2*M*NB)
						Dgebrd(m, m, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply right vectors bidiagonalizing L by Q in VT
						//                    (Workspace: need 3*M + N, prefer 3*M + N*NB)
						Dormbr('P', 'L', 'T', m, n, m, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing right
						//                    singular vectors of A in VT
						//                    (Workspace: need BDSPAC)
						Dbdsqr('U', m, n, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), s, work.Off(ie-1), vt, ldvt, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

					}

				} else if wntuo {
					//                 Path 5t(N much larger than M, JOBU='O', JOBVT='S')
					//                 M right singular vectors to be computed in VT and
					//                 M left singular vectors to be overwritten on A
					if (*lwork) >= 2*(*m)*(*m)+maxint(4*(*m), bdspac) {
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
						//                    (Workspace: need 2*M*M + 2*M, prefer 2*M*M + M + M*NB)
						Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy L to WORK(IU), zeroing out below it
						Dlacpy('L', m, m, a, lda, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, work.MatrixOff(iu+ldwrku-1, ldwrku, opts), &ldwrku)

						//                    Generate Q in A
						//                    (Workspace: need 2*M*M + 2*M, prefer 2*M*M + M + M*NB)
						Dorglq(m, n, m, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = itau
						itauq = ie + (*m)
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Bidiagonalize L in WORK(IU), copying result to
						//                    WORK(IR)
						//                    (Workspace: need 2*M*M + 4*M,
						//                                prefer 2*M*M+3*M+2*M*NB)
						Dgebrd(m, m, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('L', m, m, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr)

						//                    Generate right bidiagonalizing vectors in WORK(IU)
						//                    (Workspace: need 2*M*M + 4*M-1,
						//                                prefer 2*M*M+3*M+(M-1)*NB)
						Dorgbr('P', m, m, m, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left bidiagonalizing vectors in WORK(IR)
						//                    (Workspace: need 2*M*M + 4*M, prefer 2*M*M + 3*M + M*NB)
						Dorgbr('Q', m, m, m, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of L in WORK(IR) and computing
						//                    right singular vectors of L in WORK(IU)
						//                    (Workspace: need 2*M*M + BDSPAC)
						Dbdsqr('U', m, m, m, func() *int { y := 0; return &y }(), s, work.Off(ie-1), work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

						//                    Multiply right singular vectors of L in WORK(IU) by
						//                    Q in A, storing result in VT
						//                    (Workspace: need M*M)
						goblas.Dgemm(NoTrans, NoTrans, m, n, m, &one, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, a, lda, &zero, vt, ldvt)

						//                    Copy left singular vectors of L to A
						//                    (Workspace: need M*M)
						Dlacpy('F', m, m, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, a, lda)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*m)

						//                    Compute A=L*Q, copying result to VT
						//                    (Workspace: need 2*M, prefer M + M*NB)
						Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('U', m, n, a, lda, vt, ldvt)

						//                    Generate Q in VT
						//                    (Workspace: need 2*M, prefer M + M*NB)
						Dorglq(m, n, m, vt, ldvt, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = itau
						itauq = ie + (*m)
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Zero out above L in A
						Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, a.Off(0, 1), lda)

						//                    Bidiagonalize L in A
						//                    (Workspace: need 4*M, prefer 3*M + 2*M*NB)
						Dgebrd(m, m, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply right vectors bidiagonalizing L by Q in VT
						//                    (Workspace: need 3*M + N, prefer 3*M + N*NB)
						Dormbr('P', 'L', 'T', m, n, m, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left bidiagonalizing vectors of L in A
						//                    (Workspace: need 4*M, prefer 3*M + M*NB)
						Dorgbr('Q', m, m, m, a, lda, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, compute left
						//                    singular vectors of A in A and compute right
						//                    singular vectors of A in VT
						//                    (Workspace: need BDSPAC)
						Dbdsqr('U', m, n, m, func() *int { y := 0; return &y }(), s, work.Off(ie-1), vt, ldvt, a, lda, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

					}

				} else if wntuas {
					//                 Path 6t(N much larger than M, JOBU='S' or 'A',
					//                         JOBVT='S')
					//                 M right singular vectors to be computed in VT and
					//                 M left singular vectors to be computed in U
					if (*lwork) >= (*m)*(*m)+maxint(4*(*m), bdspac) {
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
						//                    (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
						Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy L to WORK(IU), zeroing out above it
						Dlacpy('L', m, m, a, lda, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, work.MatrixOff(iu+ldwrku-1, ldwrku, opts), &ldwrku)

						//                    Generate Q in A
						//                    (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
						Dorglq(m, n, m, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = itau
						itauq = ie + (*m)
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Bidiagonalize L in WORK(IU), copying result to U
						//                    (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
						Dgebrd(m, m, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('L', m, m, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, u, ldu)

						//                    Generate right bidiagonalizing vectors in WORK(IU)
						//                    (Workspace: need M*M + 4*M-1,
						//                                prefer M*M+3*M+(M-1)*NB)
						Dorgbr('P', m, m, m, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left bidiagonalizing vectors in U
						//                    (Workspace: need M*M + 4*M, prefer M*M + 3*M + M*NB)
						Dorgbr('Q', m, m, m, u, ldu, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of L in U and computing right
						//                    singular vectors of L in WORK(IU)
						//                    (Workspace: need M*M + BDSPAC)
						Dbdsqr('U', m, m, m, func() *int { y := 0; return &y }(), s, work.Off(ie-1), work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, u, ldu, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

						//                    Multiply right singular vectors of L in WORK(IU) by
						//                    Q in A, storing result in VT
						//                    (Workspace: need M*M)
						goblas.Dgemm(NoTrans, NoTrans, m, n, m, &one, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, a, lda, &zero, vt, ldvt)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*m)

						//                    Compute A=L*Q, copying result to VT
						//                    (Workspace: need 2*M, prefer M + M*NB)
						Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('U', m, n, a, lda, vt, ldvt)

						//                    Generate Q in VT
						//                    (Workspace: need 2*M, prefer M + M*NB)
						Dorglq(m, n, m, vt, ldvt, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy L to U, zeroing out above it
						Dlacpy('L', m, m, a, lda, u, ldu)
						Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, u.Off(0, 1), ldu)
						ie = itau
						itauq = ie + (*m)
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Bidiagonalize L in U
						//                    (Workspace: need 4*M, prefer 3*M + 2*M*NB)
						Dgebrd(m, m, u, ldu, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply right bidiagonalizing vectors in U by Q
						//                    in VT
						//                    (Workspace: need 3*M + N, prefer 3*M + N*NB)
						Dormbr('P', 'L', 'T', m, n, m, u, ldu, work.Off(itaup-1), vt, ldvt, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left bidiagonalizing vectors in U
						//                    (Workspace: need 4*M, prefer 3*M + M*NB)
						Dorgbr('Q', m, m, m, u, ldu, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in VT
						//                    (Workspace: need BDSPAC)
						Dbdsqr('U', m, n, m, func() *int { y := 0; return &y }(), s, work.Off(ie-1), vt, ldvt, u, ldu, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

					}

				}

			} else if wntva {

				if wntun {
					//                 Path 7t(N much larger than M, JOBU='N', JOBVT='A')
					//                 N right singular vectors to be computed in VT and
					//                 no left singular vectors to be computed
					if (*lwork) >= (*m)*(*m)+maxint((*n)+(*m), 4*(*m), bdspac) {
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
						//                    (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
						Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('U', m, n, a, lda, vt, ldvt)

						//                    Copy L to WORK(IR), zeroing out above it
						Dlacpy('L', m, m, a, lda, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr)
						Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, work.MatrixOff(ir+ldwrkr-1, ldwrkr, opts), &ldwrkr)

						//                    Generate Q in VT
						//                    (Workspace: need M*M + M + N, prefer M*M + M + N*NB)
						Dorglq(n, n, m, vt, ldvt, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = itau
						itauq = ie + (*m)
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Bidiagonalize L in WORK(IR)
						//                    (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
						Dgebrd(m, m, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate right bidiagonalizing vectors in WORK(IR)
						//                    (Workspace: need M*M + 4*M-1,
						//                                prefer M*M+3*M+(M-1)*NB)
						Dorgbr('P', m, m, m, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing right
						//                    singular vectors of L in WORK(IR)
						//                    (Workspace: need M*M + BDSPAC)
						Dbdsqr('U', m, m, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), s, work.Off(ie-1), work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

						//                    Multiply right singular vectors of L in WORK(IR) by
						//                    Q in VT, storing result in A
						//                    (Workspace: need M*M)
						goblas.Dgemm(NoTrans, NoTrans, m, n, m, &one, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, vt, ldvt, &zero, a, lda)

						//                    Copy right singular vectors of A from A to VT
						Dlacpy('F', m, n, a, lda, vt, ldvt)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*m)

						//                    Compute A=L*Q, copying result to VT
						//                    (Workspace: need 2*M, prefer M + M*NB)
						Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('U', m, n, a, lda, vt, ldvt)

						//                    Generate Q in VT
						//                    (Workspace: need M + N, prefer M + N*NB)
						Dorglq(n, n, m, vt, ldvt, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = itau
						itauq = ie + (*m)
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Zero out above L in A
						Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, a.Off(0, 1), lda)

						//                    Bidiagonalize L in A
						//                    (Workspace: need 4*M, prefer 3*M + 2*M*NB)
						Dgebrd(m, m, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply right bidiagonalizing vectors in A by Q
						//                    in VT
						//                    (Workspace: need 3*M + N, prefer 3*M + N*NB)
						Dormbr('P', 'L', 'T', m, n, m, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing right
						//                    singular vectors of A in VT
						//                    (Workspace: need BDSPAC)
						Dbdsqr('U', m, n, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), s, work.Off(ie-1), vt, ldvt, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

					}

				} else if wntuo {
					//                 Path 8t(N much larger than M, JOBU='O', JOBVT='A')
					//                 N right singular vectors to be computed in VT and
					//                 M left singular vectors to be overwritten on A
					if (*lwork) >= 2*(*m)*(*m)+maxint((*n)+(*m), 4*(*m), bdspac) {
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
						//                    (Workspace: need 2*M*M + 2*M, prefer 2*M*M + M + M*NB)
						Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('U', m, n, a, lda, vt, ldvt)

						//                    Generate Q in VT
						//                    (Workspace: need 2*M*M + M + N, prefer 2*M*M + M + N*NB)
						Dorglq(n, n, m, vt, ldvt, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy L to WORK(IU), zeroing out above it
						Dlacpy('L', m, m, a, lda, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, work.MatrixOff(iu+ldwrku-1, ldwrku, opts), &ldwrku)
						ie = itau
						itauq = ie + (*m)
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Bidiagonalize L in WORK(IU), copying result to
						//                    WORK(IR)
						//                    (Workspace: need 2*M*M + 4*M,
						//                                prefer 2*M*M+3*M+2*M*NB)
						Dgebrd(m, m, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('L', m, m, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr)

						//                    Generate right bidiagonalizing vectors in WORK(IU)
						//                    (Workspace: need 2*M*M + 4*M-1,
						//                                prefer 2*M*M+3*M+(M-1)*NB)
						Dorgbr('P', m, m, m, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left bidiagonalizing vectors in WORK(IR)
						//                    (Workspace: need 2*M*M + 4*M, prefer 2*M*M + 3*M + M*NB)
						Dorgbr('Q', m, m, m, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of L in WORK(IR) and computing
						//                    right singular vectors of L in WORK(IU)
						//                    (Workspace: need 2*M*M + BDSPAC)
						Dbdsqr('U', m, m, m, func() *int { y := 0; return &y }(), s, work.Off(ie-1), work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

						//                    Multiply right singular vectors of L in WORK(IU) by
						//                    Q in VT, storing result in A
						//                    (Workspace: need M*M)
						goblas.Dgemm(NoTrans, NoTrans, m, n, m, &one, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, vt, ldvt, &zero, a, lda)

						//                    Copy right singular vectors of A from A to VT
						Dlacpy('F', m, n, a, lda, vt, ldvt)

						//                    Copy left singular vectors of A from WORK(IR) to A
						Dlacpy('F', m, m, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, a, lda)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*m)

						//                    Compute A=L*Q, copying result to VT
						//                    (Workspace: need 2*M, prefer M + M*NB)
						Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('U', m, n, a, lda, vt, ldvt)

						//                    Generate Q in VT
						//                    (Workspace: need M + N, prefer M + N*NB)
						Dorglq(n, n, m, vt, ldvt, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						ie = itau
						itauq = ie + (*m)
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Zero out above L in A
						Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, a.Off(0, 1), lda)

						//                    Bidiagonalize L in A
						//                    (Workspace: need 4*M, prefer 3*M + 2*M*NB)
						Dgebrd(m, m, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply right bidiagonalizing vectors in A by Q
						//                    in VT
						//                    (Workspace: need 3*M + N, prefer 3*M + N*NB)
						Dormbr('P', 'L', 'T', m, n, m, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left bidiagonalizing vectors in A
						//                    (Workspace: need 4*M, prefer 3*M + M*NB)
						Dorgbr('Q', m, m, m, a, lda, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in A and computing right
						//                    singular vectors of A in VT
						//                    (Workspace: need BDSPAC)
						Dbdsqr('U', m, n, m, func() *int { y := 0; return &y }(), s, work.Off(ie-1), vt, ldvt, a, lda, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

					}

				} else if wntuas {
					//                 Path 9t(N much larger than M, JOBU='S' or 'A',
					//                         JOBVT='A')
					//                 N right singular vectors to be computed in VT and
					//                 M left singular vectors to be computed in U
					if (*lwork) >= (*m)*(*m)+maxint((*n)+(*m), 4*(*m), bdspac) {
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
						//                    (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
						Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('U', m, n, a, lda, vt, ldvt)

						//                    Generate Q in VT
						//                    (Workspace: need M*M + M + N, prefer M*M + M + N*NB)
						Dorglq(n, n, m, vt, ldvt, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy L to WORK(IU), zeroing out above it
						Dlacpy('L', m, m, a, lda, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku)
						Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, work.MatrixOff(iu+ldwrku-1, ldwrku, opts), &ldwrku)
						ie = itau
						itauq = ie + (*m)
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Bidiagonalize L in WORK(IU), copying result to U
						//                    (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
						Dgebrd(m, m, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('L', m, m, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, u, ldu)

						//                    Generate right bidiagonalizing vectors in WORK(IU)
						//                    (Workspace: need M*M + 4*M, prefer M*M + 3*M + (M-1)*NB)
						Dorgbr('P', m, m, m, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left bidiagonalizing vectors in U
						//                    (Workspace: need M*M + 4*M, prefer M*M + 3*M + M*NB)
						Dorgbr('Q', m, m, m, u, ldu, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of L in U and computing right
						//                    singular vectors of L in WORK(IU)
						//                    (Workspace: need M*M + BDSPAC)
						Dbdsqr('U', m, m, m, func() *int { y := 0; return &y }(), s, work.Off(ie-1), work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, u, ldu, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

						//                    Multiply right singular vectors of L in WORK(IU) by
						//                    Q in VT, storing result in A
						//                    (Workspace: need M*M)
						goblas.Dgemm(NoTrans, NoTrans, m, n, m, &one, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, vt, ldvt, &zero, a, lda)

						//                    Copy right singular vectors of A from A to VT
						Dlacpy('F', m, n, a, lda, vt, ldvt)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + (*m)

						//                    Compute A=L*Q, copying result to VT
						//                    (Workspace: need 2*M, prefer M + M*NB)
						Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						Dlacpy('U', m, n, a, lda, vt, ldvt)

						//                    Generate Q in VT
						//                    (Workspace: need M + N, prefer M + N*NB)
						Dorglq(n, n, m, vt, ldvt, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Copy L to U, zeroing out above it
						Dlacpy('L', m, m, a, lda, u, ldu)
						Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, u.Off(0, 1), ldu)
						ie = itau
						itauq = ie + (*m)
						itaup = itauq + (*m)
						iwork = itaup + (*m)

						//                    Bidiagonalize L in U
						//                    (Workspace: need 4*M, prefer 3*M + 2*M*NB)
						Dgebrd(m, m, u, ldu, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Multiply right bidiagonalizing vectors in U by Q
						//                    in VT
						//                    (Workspace: need 3*M + N, prefer 3*M + N*NB)
						Dormbr('P', 'L', 'T', m, n, m, u, ldu, work.Off(itaup-1), vt, ldvt, work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)

						//                    Generate left bidiagonalizing vectors in U
						//                    (Workspace: need 4*M, prefer 3*M + M*NB)
						Dorgbr('Q', m, m, m, u, ldu, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
						iwork = ie + (*m)

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in VT
						//                    (Workspace: need BDSPAC)
						Dbdsqr('U', m, n, m, func() *int { y := 0; return &y }(), s, work.Off(ie-1), vt, ldvt, u, ldu, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)

					}

				}

			}

		} else {
			//           N .LT. MNTHR
			//
			//           Path 10t(N greater than M, but not much larger)
			//           Reduce to bidiagonal form without LQ decomposition
			ie = 1
			itauq = ie + (*m)
			itaup = itauq + (*m)
			iwork = itaup + (*m)

			//           Bidiagonalize A
			//           (Workspace: need 3*M + N, prefer 3*M + (M + N)*NB)
			Dgebrd(m, n, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			if wntuas {
				//              If left singular vectors desired in U, copy result to U
				//              and generate left bidiagonalizing vectors in U
				//              (Workspace: need 4*M-1, prefer 3*M + (M-1)*NB)
				Dlacpy('L', m, m, a, lda, u, ldu)
				Dorgbr('Q', m, m, n, u, ldu, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			}
			if wntvas {
				//              If right singular vectors desired in VT, copy result to
				//              VT and generate right bidiagonalizing vectors in VT
				//              (Workspace: need 3*M + NRVT, prefer 3*M + NRVT*NB)
				Dlacpy('U', m, n, a, lda, vt, ldvt)
				if wntva {
					nrvt = (*n)
				}
				if wntvs {
					nrvt = (*m)
				}
				Dorgbr('P', &nrvt, n, m, vt, ldvt, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			}
			if wntuo {
				//              If left singular vectors desired in A, generate left
				//              bidiagonalizing vectors in A
				//              (Workspace: need 4*M-1, prefer 3*M + (M-1)*NB)
				Dorgbr('Q', m, m, n, a, lda, work.Off(itauq-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			}
			if wntvo {
				//              If right singular vectors desired in A, generate right
				//              bidiagonalizing vectors in A
				//              (Workspace: need 4*M, prefer 3*M + M*NB)
				Dorgbr('P', m, n, m, a, lda, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), &ierr)
			}
			iwork = ie + (*m)
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
				//              (Workspace: need BDSPAC)
				Dbdsqr('L', m, &ncvt, &nru, func() *int { y := 0; return &y }(), s, work.Off(ie-1), vt, ldvt, u, ldu, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)
			} else if (!wntuo) && wntvo {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in U and computing right singular
				//              vectors in A
				//              (Workspace: need BDSPAC)
				Dbdsqr('L', m, &ncvt, &nru, func() *int { y := 0; return &y }(), s, work.Off(ie-1), a, lda, u, ldu, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)
			} else {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in A and computing right singular
				//              vectors in VT
				//              (Workspace: need BDSPAC)
				Dbdsqr('L', m, &ncvt, &nru, func() *int { y := 0; return &y }(), s, work.Off(ie-1), vt, ldvt, a, lda, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), work.Off(iwork-1), info)
			}

		}

	}

	//     If DBDSQR failed to converge, copy unconverged superdiagonals
	//     to WORK( 2:MINMN )
	if (*info) != 0 {
		if ie > 2 {
			for i = 1; i <= minmn-1; i++ {
				work.Set(i+1-1, work.Get(i+ie-1-1))
			}
		}
		if ie < 2 {
			for i = minmn - 1; i >= 1; i-- {
				work.Set(i+1-1, work.Get(i+ie-1-1))
			}
		}
	}

	//     Undo scaling if necessary
	if iscl == 1 {
		if anrm > bignum {
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bignum, &anrm, &minmn, func() *int { y := 1; return &y }(), s.Matrix(minmn, opts), &minmn, &ierr)
		}
		if (*info) != 0 && anrm > bignum {
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bignum, &anrm, toPtr(minmn-1), func() *int { y := 1; return &y }(), work.MatrixOff(1, minmn, opts), &minmn, &ierr)
		}
		if anrm < smlnum {
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &smlnum, &anrm, &minmn, func() *int { y := 1; return &y }(), s.Matrix(minmn, opts), &minmn, &ierr)
		}
		if (*info) != 0 && anrm < smlnum {
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &smlnum, &anrm, toPtr(minmn-1), func() *int { y := 1; return &y }(), work.MatrixOff(1, minmn, opts), &minmn, &ierr)
		}
	}

	//     Return optimal workspace in WORK(1)
	work.Set(0, float64(maxwrk))
}
