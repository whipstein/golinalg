package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgesdd computes the singular value decomposition (SVD) of a real
// M-by-N matrix A, optionally computing the left and right singular
// vectors.  If singular vectors are desired, it uses a
// divide-and-conquer algorithm.
//
// The SVD is written
//
//      A = U * SIGMA * transpose(V)
//
// where SIGMA is an M-by-N matrix which is zero except for its
// minint(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
// V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
// are the singular values of A; they are real and non-negative, and
// are returned in descending order.  The first minint(m,n) columns of
// U and V are the left and right singular vectors of A.
//
// Note that the routine returns VT = V**T, not V.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Dgesdd(jobz byte, m, n *int, a *mat.Matrix, lda *int, s *mat.Vector, u *mat.Matrix, ldu *int, vt *mat.Matrix, ldvt *int, work *mat.Vector, lwork *int, iwork *[]int, info *int) {
	var lquery, wntqa, wntqas, wntqn, wntqo, wntqs bool
	var anrm, bignum, eps, one, smlnum, zero float64
	var bdspac, blk, chunk, i, ie, ierr, il, ir, iscl, itau, itaup, itauq, iu, ivt, ldwkvt, ldwrkl, ldwrkr, ldwrku, lworkDgebrdMm, lworkDgebrdMn, lworkDgebrdNn, lworkDgelqfMn, lworkDgeqrfMn, lworkDorglqMn, lworkDorglqNn, lworkDorgqrMm, lworkDorgqrMn, lworkDormbrPrtMm, lworkDormbrPrtMn, lworkDormbrPrtNn, lworkDormbrQlnMm, lworkDormbrQlnMn, lworkDormbrQlnNn, maxwrk, minmn, minwrk, mnthr, nwork, wrkbl int
	var err error
	_ = err

	dum := vf(1)
	idum := make([]int, 1)

	zero = 0.0
	one = 1.0

	//     Test the input arguments
	(*info) = 0
	minmn = minint(*m, *n)
	wntqa = jobz == 'A'
	wntqs = jobz == 'S'
	wntqas = wntqa || wntqs
	wntqo = jobz == 'O'
	wntqn = jobz == 'N'
	lquery = ((*lwork) == -1)

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
	//       NB refers to the optimal block size for the immediately
	//       following subroutine, as returned by ILAENV.
	if (*info) == 0 {
		minwrk = 1
		maxwrk = 1
		bdspac = 0
		mnthr = int(minmn * 11.0 / 6.0)
		if (*m) >= (*n) && minmn > 0 {
			//           Compute space needed for DBDSDC
			if wntqn {
				//              dbdsdc needs only 4*N (or 6*N for uplo=L for LAPACK <= 3.6)
				//              keep 7*N for backwards compatibility.
				bdspac = 7 * (*n)
			} else {
				bdspac = 3*(*n)*(*n) + 4*(*n)
			}

			//           Compute space preferred for each routine
			Dgebrd(m, n, dum.Matrix(*m, opts), m, dum, dum, dum, dum, dum, toPtr(-1), &ierr)
			lworkDgebrdMn = int(dum.Get(0))

			Dgebrd(n, n, dum.Matrix(*m, opts), n, dum, dum, dum, dum, dum, toPtr(-1), &ierr)
			lworkDgebrdNn = int(dum.Get(0))

			Dgeqrf(m, n, dum.Matrix(*m, opts), m, dum, dum, toPtr(-1), &ierr)
			lworkDgeqrfMn = int(dum.Get(0))

			Dorgbr('Q', n, n, n, dum.Matrix(*m, opts), n, dum, dum, toPtr(-1), &ierr)
			// lworkDorgbrQNn = int(dum.Get(0))

			Dorgqr(m, m, n, dum.Matrix(*m, opts), m, dum, dum, toPtr(-1), &ierr)
			lworkDorgqrMm = int(dum.Get(0))

			Dorgqr(m, n, n, dum.Matrix(*m, opts), m, dum, dum, toPtr(-1), &ierr)
			lworkDorgqrMn = int(dum.Get(0))

			Dormbr('P', 'R', 'T', n, n, n, dum.Matrix(*n, opts), n, dum, dum.Matrix(*n, opts), n, dum, toPtr(-1), &ierr)
			lworkDormbrPrtNn = int(dum.Get(0))

			Dormbr('Q', 'L', 'N', n, n, n, dum.Matrix(*n, opts), n, dum, dum.Matrix(*n, opts), n, dum, toPtr(-1), &ierr)
			lworkDormbrQlnNn = int(dum.Get(0))

			Dormbr('Q', 'L', 'N', m, n, n, dum.Matrix(*n, opts), m, dum, dum.Matrix(*n, opts), m, dum, toPtr(-1), &ierr)
			lworkDormbrQlnMn = int(dum.Get(0))

			Dormbr('Q', 'L', 'N', m, m, n, dum.Matrix(*n, opts), m, dum, dum.Matrix(*n, opts), m, dum, toPtr(-1), &ierr)
			lworkDormbrQlnMm = int(dum.Get(0))

			if (*m) >= mnthr {
				if wntqn {
					//                 Path 1 (M >> N, JOBZ='N')
					wrkbl = (*n) + lworkDgeqrfMn
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDgebrdNn)
					maxwrk = maxint(wrkbl, bdspac+(*n))
					minwrk = bdspac + (*n)
				} else if wntqo {
					//                 Path 2 (M >> N, JOBZ='O')
					wrkbl = (*n) + lworkDgeqrfMn
					wrkbl = maxint(wrkbl, (*n)+lworkDorgqrMn)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDgebrdNn)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDormbrQlnNn)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDormbrPrtNn)
					wrkbl = maxint(wrkbl, 3*(*n)+bdspac)
					maxwrk = wrkbl + 2*(*n)*(*n)
					minwrk = bdspac + 2*(*n)*(*n) + 3*(*n)
				} else if wntqs {
					//                 Path 3 (M >> N, JOBZ='S')
					wrkbl = (*n) + lworkDgeqrfMn
					wrkbl = maxint(wrkbl, (*n)+lworkDorgqrMn)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDgebrdNn)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDormbrQlnNn)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDormbrPrtNn)
					wrkbl = maxint(wrkbl, 3*(*n)+bdspac)
					maxwrk = wrkbl + (*n)*(*n)
					minwrk = bdspac + (*n)*(*n) + 3*(*n)
				} else if wntqa {
					//                 Path 4 (M >> N, JOBZ='A')
					wrkbl = (*n) + lworkDgeqrfMn
					wrkbl = maxint(wrkbl, (*n)+lworkDorgqrMm)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDgebrdNn)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDormbrQlnNn)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDormbrPrtNn)
					wrkbl = maxint(wrkbl, 3*(*n)+bdspac)
					maxwrk = wrkbl + (*n)*(*n)
					minwrk = (*n)*(*n) + maxint(3*(*n)+bdspac, (*n)+(*m))
				}
			} else {
				//              Path 5 (M >= N, but not much larger)
				wrkbl = 3*(*n) + lworkDgebrdMn
				if wntqn {
					//                 Path 5n (M >= N, jobz='N')
					maxwrk = maxint(wrkbl, 3*(*n)+bdspac)
					minwrk = 3*(*n) + maxint(*m, bdspac)
				} else if wntqo {
					//                 Path 5o (M >= N, jobz='O')
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDormbrPrtNn)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDormbrQlnMn)
					wrkbl = maxint(wrkbl, 3*(*n)+bdspac)
					maxwrk = wrkbl + (*m)*(*n)
					minwrk = 3*(*n) + maxint(*m, (*n)*(*n)+bdspac)
				} else if wntqs {
					//                 Path 5s (M >= N, jobz='S')
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDormbrQlnMn)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDormbrPrtNn)
					maxwrk = maxint(wrkbl, 3*(*n)+bdspac)
					minwrk = 3*(*n) + maxint(*m, bdspac)
				} else if wntqa {
					//                 Path 5a (M >= N, jobz='A')
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDormbrQlnMm)
					wrkbl = maxint(wrkbl, 3*(*n)+lworkDormbrPrtNn)
					maxwrk = maxint(wrkbl, 3*(*n)+bdspac)
					minwrk = 3*(*n) + maxint(*m, bdspac)
				}
			}
		} else if minmn > 0 {
			//           Compute space needed for DBDSDC
			if wntqn {
				//              dbdsdc needs only 4*N (or 6*N for uplo=L for LAPACK <= 3.6)
				//              keep 7*N for backwards compatibility.
				bdspac = 7 * (*m)
			} else {
				bdspac = 3*(*m)*(*m) + 4*(*m)
			}

			//           Compute space preferred for each routine
			Dgebrd(m, n, dum.Matrix(*m, opts), m, dum, dum, dum, dum, dum, toPtr(-1), &ierr)
			lworkDgebrdMn = int(dum.Get(0))

			Dgebrd(m, m, a, m, s, dum, dum, dum, dum, toPtr(-1), &ierr)
			lworkDgebrdMm = int(dum.Get(0))

			Dgelqf(m, n, a, m, dum, dum, toPtr(-1), &ierr)
			lworkDgelqfMn = int(dum.Get(0))

			Dorglq(n, n, m, dum.Matrix(*n, opts), n, dum, dum, toPtr(-1), &ierr)
			lworkDorglqNn = int(dum.Get(0))

			Dorglq(m, n, m, a, m, dum, dum, toPtr(-1), &ierr)
			lworkDorglqMn = int(dum.Get(0))

			Dorgbr('P', m, m, m, a, n, dum, dum, toPtr(-1), &ierr)
			// lworkDorgbrPMm = int(dum.Get(0))

			Dormbr('P', 'R', 'T', m, m, m, dum.Matrix(*m, opts), m, dum, dum.Matrix(*m, opts), m, dum, toPtr(-1), &ierr)
			lworkDormbrPrtMm = int(dum.Get(0))

			Dormbr('P', 'R', 'T', m, n, m, dum.Matrix(*m, opts), m, dum, dum.Matrix(*m, opts), m, dum, toPtr(-1), &ierr)
			lworkDormbrPrtMn = int(dum.Get(0))

			Dormbr('P', 'R', 'T', n, n, m, dum.Matrix(*n, opts), n, dum, dum.Matrix(*n, opts), n, dum, toPtr(-1), &ierr)
			lworkDormbrPrtNn = int(dum.Get(0))

			Dormbr('Q', 'L', 'N', m, m, m, dum.Matrix(*m, opts), m, dum, dum.Matrix(*m, opts), m, dum, toPtr(-1), &ierr)
			lworkDormbrQlnMm = int(dum.Get(0))

			if (*n) >= mnthr {
				if wntqn {
					//                 Path 1t (N >> M, JOBZ='N')
					wrkbl = (*m) + lworkDgelqfMn
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDgebrdMm)
					maxwrk = maxint(wrkbl, bdspac+(*m))
					minwrk = bdspac + (*m)
				} else if wntqo {
					//                 Path 2t (N >> M, JOBZ='O')
					wrkbl = (*m) + lworkDgelqfMn
					wrkbl = maxint(wrkbl, (*m)+lworkDorglqMn)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDgebrdMm)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDormbrQlnMm)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDormbrPrtMm)
					wrkbl = maxint(wrkbl, 3*(*m)+bdspac)
					maxwrk = wrkbl + 2*(*m)*(*m)
					minwrk = bdspac + 2*(*m)*(*m) + 3*(*m)
				} else if wntqs {
					//                 Path 3t (N >> M, JOBZ='S')
					wrkbl = (*m) + lworkDgelqfMn
					wrkbl = maxint(wrkbl, (*m)+lworkDorglqMn)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDgebrdMm)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDormbrQlnMm)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDormbrPrtMm)
					wrkbl = maxint(wrkbl, 3*(*m)+bdspac)
					maxwrk = wrkbl + (*m)*(*m)
					minwrk = bdspac + (*m)*(*m) + 3*(*m)
				} else if wntqa {
					//                 Path 4t (N >> M, JOBZ='A')
					wrkbl = (*m) + lworkDgelqfMn
					wrkbl = maxint(wrkbl, (*m)+lworkDorglqNn)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDgebrdMm)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDormbrQlnMm)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDormbrPrtMm)
					wrkbl = maxint(wrkbl, 3*(*m)+bdspac)
					maxwrk = wrkbl + (*m)*(*m)
					minwrk = (*m)*(*m) + maxint(3*(*m)+bdspac, (*m)+(*n))
				}
			} else {
				//              Path 5t (N > M, but not much larger)
				wrkbl = 3*(*m) + lworkDgebrdMn
				if wntqn {
					//                 Path 5tn (N > M, jobz='N')
					maxwrk = maxint(wrkbl, 3*(*m)+bdspac)
					minwrk = 3*(*m) + maxint(*n, bdspac)
				} else if wntqo {
					//                 Path 5to (N > M, jobz='O')
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDormbrQlnMm)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDormbrPrtMn)
					wrkbl = maxint(wrkbl, 3*(*m)+bdspac)
					maxwrk = wrkbl + (*m)*(*n)
					minwrk = 3*(*m) + maxint(*n, (*m)*(*m)+bdspac)
				} else if wntqs {
					//                 Path 5ts (N > M, jobz='S')
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDormbrQlnMm)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDormbrPrtMn)
					maxwrk = maxint(wrkbl, 3*(*m)+bdspac)
					minwrk = 3*(*m) + maxint(*n, bdspac)
				} else if wntqa {
					//                 Path 5ta (N > M, jobz='A')
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDormbrQlnMm)
					wrkbl = maxint(wrkbl, 3*(*m)+lworkDormbrPrtNn)
					maxwrk = maxint(wrkbl, 3*(*m)+bdspac)
					minwrk = 3*(*m) + maxint(*n, bdspac)
				}
			}
		}
		maxwrk = maxint(maxwrk, minwrk)
		work.Set(0, float64(maxwrk))

		if (*lwork) < minwrk && !lquery {
			(*info) = -12
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGESDD"), -(*info))
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
		Dlascl('G', toPtr(0), toPtr(0), &anrm, &smlnum, m, n, a, lda, &ierr)
	} else if anrm > bignum {
		iscl = 1
		Dlascl('G', toPtr(0), toPtr(0), &anrm, &bignum, m, n, a, lda, &ierr)
	}

	if (*m) >= (*n) {
		//        A has at least as many rows as columns. If A has sufficiently
		//        more rows than columns, first reduce using the QR
		//        decomposition (if sufficient workspace available)
		if (*m) >= mnthr {

			if wntqn {
				//              Path 1 (M >> N, JOBZ='N')
				//              No singular vectors to be computed
				itau = 1
				nwork = itau + (*n)

				//              Compute A=Q*R
				//              Workspace: need   N [tau] + N    [work]
				//              Workspace: prefer N [tau] + N*NB [work]
				Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Zero out below R
				Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, a.Off(1, 0), lda)
				ie = 1
				itauq = ie + (*n)
				itaup = itauq + (*n)
				nwork = itaup + (*n)

				//              Bidiagonalize R in A
				//              Workspace: need   3*N [e, tauq, taup] + N      [work]
				//              Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [work]
				Dgebrd(n, n, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				nwork = ie + (*n)

				//              Perform bidiagonal SVD, computing singular values only
				//              Workspace: need   N [e] + BDSPAC
				Dbdsdc('U', 'N', n, s, work.Off(ie-1), dum.Matrix(1, opts), toPtr(1), dum.Matrix(1, opts), toPtr(1), dum, &idum, work.Off(nwork-1), iwork, info)

			} else if wntqo {
				//              Path 2 (M >> N, JOBZ = 'O')
				//              N left singular vectors to be overwritten on A and
				//              N right singular vectors to be computed in VT
				ir = 1

				//              WORK(IR) is LDWRKR by N
				if (*lwork) >= (*lda)*(*n)+(*n)*(*n)+3*(*n)+bdspac {
					ldwrkr = (*lda)
				} else {
					ldwrkr = ((*lwork) - (*n)*(*n) - 3*(*n) - bdspac) / (*n)
				}
				itau = ir + ldwrkr*(*n)
				nwork = itau + (*n)

				//              Compute A=Q*R
				//              Workspace: need   N*N [R] + N [tau] + N    [work]
				//              Workspace: prefer N*N [R] + N [tau] + N*NB [work]
				Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy R to WORK(IR), zeroing out below it
				Dlacpy('U', n, n, a, lda, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr)
				Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, work.MatrixOff(ir+1-1, ldwrkr, opts), &ldwrkr)

				//              Generate Q in A
				//              Workspace: need   N*N [R] + N [tau] + N    [work]
				//              Workspace: prefer N*N [R] + N [tau] + N*NB [work]
				Dorgqr(m, n, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				ie = itau
				itauq = ie + (*n)
				itaup = itauq + (*n)
				nwork = itaup + (*n)

				//              Bidiagonalize R in WORK(IR)
				//              Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N      [work]
				//              Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + 2*N*NB [work]
				Dgebrd(n, n, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              WORK(IU) is N by N
				iu = nwork
				nwork = iu + (*n)*(*n)

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in WORK(IU) and computing right
				//              singular vectors of bidiagonal matrix in VT
				//              Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N*N [U] + BDSPAC
				Dbdsdc('U', 'I', n, s, work.Off(ie-1), work.MatrixOff(iu-1, *n, opts), n, vt, ldvt, dum, &idum, work.Off(nwork-1), iwork, info)

				//              Overwrite WORK(IU) by left singular vectors of R
				//              and VT by right singular vectors of R
				//              Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N*N [U] + N    [work]
				//              Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*N [U] + N*NB [work]
				Dormbr('Q', 'L', 'N', n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itauq-1), work.MatrixOff(iu-1, *n, opts), n, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				Dormbr('P', 'R', 'T', n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itaup-1), vt, ldvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Multiply Q in A by left singular vectors of R in
				//              WORK(IU), storing result in WORK(IR) and copying to A
				//              Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N*N [U]
				//              Workspace: prefer M*N [R] + 3*N [e, tauq, taup] + N*N [U]
				for i = 1; i <= (*m); i += ldwrkr {
					chunk = minint((*m)-i+1, ldwrkr)
					err = goblas.Dgemm(NoTrans, NoTrans, chunk, *n, *n, one, a.Off(i-1, 0), *lda, work.MatrixOff(iu-1, *n, opts), *n, zero, work.MatrixOff(ir-1, ldwrkr, opts), ldwrkr)
					Dlacpy('F', &chunk, n, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, a.Off(i-1, 0), lda)
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
				//              Workspace: need   N*N [R] + N [tau] + N    [work]
				//              Workspace: prefer N*N [R] + N [tau] + N*NB [work]
				Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy R to WORK(IR), zeroing out below it
				Dlacpy('U', n, n, a, lda, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr)
				Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, work.MatrixOff(ir+1-1, ldwrkr, opts), &ldwrkr)

				//              Generate Q in A
				//              Workspace: need   N*N [R] + N [tau] + N    [work]
				//              Workspace: prefer N*N [R] + N [tau] + N*NB [work]
				Dorgqr(m, n, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				ie = itau
				itauq = ie + (*n)
				itaup = itauq + (*n)
				nwork = itaup + (*n)

				//              Bidiagonalize R in WORK(IR)
				//              Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N      [work]
				//              Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + 2*N*NB [work]
				Dgebrd(n, n, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagoal matrix in U and computing right singular
				//              vectors of bidiagonal matrix in VT
				//              Workspace: need   N*N [R] + 3*N [e, tauq, taup] + BDSPAC
				Dbdsdc('U', 'I', n, s, work.Off(ie-1), u, ldu, vt, ldvt, dum, &idum, work.Off(nwork-1), iwork, info)

				//              Overwrite U by left singular vectors of R and VT
				//              by right singular vectors of R
				//              Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N    [work]
				//              Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*NB [work]
				Dormbr('Q', 'L', 'N', n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itauq-1), u, ldu, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				Dormbr('P', 'R', 'T', n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, work.Off(itaup-1), vt, ldvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Multiply Q in A by left singular vectors of R in
				//              WORK(IR), storing result in U
				//              Workspace: need   N*N [R]
				Dlacpy('F', n, n, u, ldu, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr)
				err = goblas.Dgemm(NoTrans, NoTrans, *m, *n, *n, one, a, *lda, work.MatrixOff(ir-1, ldwrkr, opts), ldwrkr, zero, u, *ldu)

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
				//              Workspace: need   N*N [U] + N [tau] + N    [work]
				//              Workspace: prefer N*N [U] + N [tau] + N*NB [work]
				Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				Dlacpy('L', m, n, a, lda, u, ldu)

				//              Generate Q in U
				//              Workspace: need   N*N [U] + N [tau] + M    [work]
				Dorgqr(m, m, n, u, ldu, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Produce R in A, zeroing out other entries
				Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, a.Off(1, 0), lda)
				ie = itau
				itauq = ie + (*n)
				itaup = itauq + (*n)
				nwork = itaup + (*n)

				//              Bidiagonalize R in A
				//              Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N      [work]
				//              Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + 2*N*NB [work]
				Dgebrd(n, n, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in WORK(IU) and computing right
				//              singular vectors of bidiagonal matrix in VT
				//              Workspace: need   N*N [U] + 3*N [e, tauq, taup] + BDSPAC
				Dbdsdc('U', 'I', n, s, work.Off(ie-1), work.MatrixOff(iu-1, *n, opts), n, vt, ldvt, dum, &idum, work.Off(nwork-1), iwork, info)

				//              Overwrite WORK(IU) by left singular vectors of R and VT
				//              by right singular vectors of R
				//              Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N    [work]
				//              Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + N*NB [work]
				Dormbr('Q', 'L', 'N', n, n, n, a, lda, work.Off(itauq-1), work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				Dormbr('P', 'R', 'T', n, n, n, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Multiply Q in U by left singular vectors of R in
				//              WORK(IU), storing result in A
				//              Workspace: need   N*N [U]
				err = goblas.Dgemm(NoTrans, NoTrans, *m, *n, *n, one, u, *ldu, work.MatrixOff(iu-1, ldwrku, opts), ldwrku, zero, a, *lda)

				//              Copy left singular vectors of A from A to U
				Dlacpy('F', m, n, a, lda, u, ldu)

			}

		} else {
			//           M .LT. MNTHR
			//
			//           Path 5 (M >= N, but not much larger)
			//           Reduce to bidiagonal form without QR decomposition

			ie = 1
			itauq = ie + (*n)
			itaup = itauq + (*n)
			nwork = itaup + (*n)

			//           Bidiagonalize A
			//           Workspace: need   3*N [e, tauq, taup] + M        [work]
			//           Workspace: prefer 3*N [e, tauq, taup] + (M+N)*NB [work]
			Dgebrd(m, n, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
			if wntqn {
				//              Path 5n (M >= N, JOBZ='N')
				//              Perform bidiagonal SVD, only computing singular values
				//              Workspace: need   3*N [e, tauq, taup] + BDSPAC
				Dbdsdc('U', 'N', n, s, work.Off(ie-1), dum.Matrix(1, opts), toPtr(1), dum.Matrix(1, opts), toPtr(1), dum, &idum, work.Off(nwork-1), iwork, info)
			} else if wntqo {
				//              Path 5o (M >= N, JOBZ='O')
				iu = nwork
				if (*lwork) >= (*m)*(*n)+3*(*n)+bdspac {
					//                 WORK( IU ) is M by N
					ldwrku = (*m)
					nwork = iu + ldwrku*(*n)
					Dlaset('F', m, n, &zero, &zero, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku)
					//                 IR is unused; silence compile warnings
					ir = -1
				} else {
					//                 WORK( IU ) is N by N
					ldwrku = (*n)
					nwork = iu + ldwrku*(*n)

					//                 WORK(IR) is LDWRKR by N
					ir = nwork
					ldwrkr = ((*lwork) - (*n)*(*n) - 3*(*n)) / (*n)
				}
				nwork = iu + ldwrku*(*n)

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in WORK(IU) and computing right
				//              singular vectors of bidiagonal matrix in VT
				//              Workspace: need   3*N [e, tauq, taup] + N*N [U] + BDSPAC
				Dbdsdc('U', 'I', n, s, work.Off(ie-1), work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, vt, ldvt, dum, &idum, work.Off(nwork-1), iwork, info)

				//              Overwrite VT by right singular vectors of A
				//              Workspace: need   3*N [e, tauq, taup] + N*N [U] + N    [work]
				//              Workspace: prefer 3*N [e, tauq, taup] + N*N [U] + N*NB [work]
				Dormbr('P', 'R', 'T', n, n, n, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				if (*lwork) >= (*m)*(*n)+3*(*n)+bdspac {
					//                 Path 5o-fast
					//                 Overwrite WORK(IU) by left singular vectors of A
					//                 Workspace: need   3*N [e, tauq, taup] + M*N [U] + N    [work]
					//                 Workspace: prefer 3*N [e, tauq, taup] + M*N [U] + N*NB [work]
					Dormbr('Q', 'L', 'N', m, n, n, a, lda, work.Off(itauq-1), work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

					//                 Copy left singular vectors of A from WORK(IU) to A
					Dlacpy('F', m, n, work.MatrixOff(iu-1, ldwrku, opts), &ldwrku, a, lda)
				} else {
					//                 Path 5o-slow
					//                 Generate Q in A
					//                 Workspace: need   3*N [e, tauq, taup] + N*N [U] + N    [work]
					//                 Workspace: prefer 3*N [e, tauq, taup] + N*N [U] + N*NB [work]
					Dorgbr('Q', m, n, n, a, lda, work.Off(itauq-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

					//                 Multiply Q in A by left singular vectors of
					//                 bidiagonal matrix in WORK(IU), storing result in
					//                 WORK(IR) and copying to A
					//                 Workspace: need   3*N [e, tauq, taup] + N*N [U] + NB*N [R]
					//                 Workspace: prefer 3*N [e, tauq, taup] + N*N [U] + M*N  [R]
					for i = 1; i <= (*m); i += ldwrkr {
						chunk = minint((*m)-i+1, ldwrkr)
						err = goblas.Dgemm(NoTrans, NoTrans, chunk, *n, *n, one, a.Off(i-1, 0), *lda, work.MatrixOff(iu-1, ldwrku, opts), ldwrku, zero, work.MatrixOff(ir-1, ldwrkr, opts), ldwrkr)
						Dlacpy('F', &chunk, n, work.MatrixOff(ir-1, ldwrkr, opts), &ldwrkr, a.Off(i-1, 0), lda)
					}
				}

			} else if wntqs {
				//              Path 5s (M >= N, JOBZ='S')
				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in U and computing right singular
				//              vectors of bidiagonal matrix in VT
				//              Workspace: need   3*N [e, tauq, taup] + BDSPAC
				Dlaset('F', m, n, &zero, &zero, u, ldu)
				Dbdsdc('U', 'I', n, s, work.Off(ie-1), u, ldu, vt, ldvt, dum, &idum, work.Off(nwork-1), iwork, info)

				//              Overwrite U by left singular vectors of A and VT
				//              by right singular vectors of A
				//              Workspace: need   3*N [e, tauq, taup] + N    [work]
				//              Workspace: prefer 3*N [e, tauq, taup] + N*NB [work]
				Dormbr('Q', 'L', 'N', m, n, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				Dormbr('P', 'R', 'T', n, n, n, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
			} else if wntqa {
				//              Path 5a (M >= N, JOBZ='A')
				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in U and computing right singular
				//              vectors of bidiagonal matrix in VT
				//              Workspace: need   3*N [e, tauq, taup] + BDSPAC
				Dlaset('F', m, m, &zero, &zero, u, ldu)
				Dbdsdc('U', 'I', n, s, work.Off(ie-1), u, ldu, vt, ldvt, dum, &idum, work.Off(nwork-1), iwork, info)

				//              Set the right corner of U to identity matrix
				if (*m) > (*n) {
					Dlaset('F', toPtr((*m)-(*n)), toPtr((*m)-(*n)), &zero, &one, u.Off((*n)+1-1, (*n)+1-1), ldu)
				}

				//              Overwrite U by left singular vectors of A and VT
				//              by right singular vectors of A
				//              Workspace: need   3*N [e, tauq, taup] + M    [work]
				//              Workspace: prefer 3*N [e, tauq, taup] + M*NB [work]
				Dormbr('Q', 'L', 'N', m, m, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				Dormbr('P', 'R', 'T', n, n, m, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
			}

		}

	} else {
		//        A has more columns than rows. If A has sufficiently more
		//        columns than rows, first reduce using the LQ decomposition (if
		//        sufficient workspace available)
		if (*n) >= mnthr {

			if wntqn {
				//              Path 1t (N >> M, JOBZ='N')
				//              No singular vectors to be computed
				itau = 1
				nwork = itau + (*m)

				//              Compute A=L*Q
				//              Workspace: need   M [tau] + M [work]
				//              Workspace: prefer M [tau] + M*NB [work]
				Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Zero out above L
				Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, a.Off(0, 1), lda)
				ie = 1
				itauq = ie + (*m)
				itaup = itauq + (*m)
				nwork = itaup + (*m)

				//              Bidiagonalize L in A
				//              Workspace: need   3*M [e, tauq, taup] + M      [work]
				//              Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [work]
				Dgebrd(m, m, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				nwork = ie + (*m)

				//              Perform bidiagonal SVD, computing singular values only
				//              Workspace: need   M [e] + BDSPAC
				Dbdsdc('U', 'N', m, s, work.Off(ie-1), dum.Matrix(1, opts), toPtr(1), dum.Matrix(1, opts), toPtr(1), dum, &idum, work.Off(nwork-1), iwork, info)

			} else if wntqo {
				//              Path 2t (N >> M, JOBZ='O')
				//              M right singular vectors to be overwritten on A and
				//              M left singular vectors to be computed in U
				ivt = 1

				//              WORK(IVT) is M by M
				//              WORK(IL)  is M by M; it is later resized to M by chunk for gemm
				il = ivt + (*m)*(*m)
				if (*lwork) >= (*m)*(*n)+(*m)*(*m)+3*(*m)+bdspac {
					ldwrkl = (*m)
					chunk = (*n)
				} else {
					ldwrkl = (*m)
					chunk = ((*lwork) - (*m)*(*m)) / (*m)
				}
				itau = il + ldwrkl*(*m)
				nwork = itau + (*m)

				//              Compute A=L*Q
				//              Workspace: need   M*M [VT] + M*M [L] + M [tau] + M    [work]
				//              Workspace: prefer M*M [VT] + M*M [L] + M [tau] + M*NB [work]
				Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy L to WORK(IL), zeroing about above it
				Dlacpy('L', m, m, a, lda, work.MatrixOff(il-1, ldwrkl, opts), &ldwrkl)
				Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, work.MatrixOff(il+ldwrkl-1, ldwrkl, opts), &ldwrkl)

				//              Generate Q in A
				//              Workspace: need   M*M [VT] + M*M [L] + M [tau] + M    [work]
				//              Workspace: prefer M*M [VT] + M*M [L] + M [tau] + M*NB [work]
				Dorglq(m, n, m, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				ie = itau
				itauq = ie + (*m)
				itaup = itauq + (*m)
				nwork = itaup + (*m)

				//              Bidiagonalize L in WORK(IL)
				//              Workspace: need   M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + M      [work]
				//              Workspace: prefer M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + 2*M*NB [work]
				Dgebrd(m, m, work.MatrixOff(il-1, ldwrkl, opts), &ldwrkl, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in U, and computing right singular
				//              vectors of bidiagonal matrix in WORK(IVT)
				//              Workspace: need   M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + BDSPAC
				Dbdsdc('U', 'I', m, s, work.Off(ie-1), u, ldu, work.MatrixOff(ivt-1, *m, opts), m, dum, &idum, work.Off(nwork-1), iwork, info)

				//              Overwrite U by left singular vectors of L and WORK(IVT)
				//              by right singular vectors of L
				//              Workspace: need   M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + M    [work]
				//              Workspace: prefer M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + M*NB [work]
				Dormbr('Q', 'L', 'N', m, m, m, work.MatrixOff(il-1, ldwrkl, opts), &ldwrkl, work.Off(itauq-1), u, ldu, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				Dormbr('P', 'R', 'T', m, m, m, work.MatrixOff(il-1, ldwrkl, opts), &ldwrkl, work.Off(itaup-1), work.MatrixOff(ivt-1, *m, opts), m, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Multiply right singular vectors of L in WORK(IVT) by Q
				//              in A, storing result in WORK(IL) and copying to A
				//              Workspace: need   M*M [VT] + M*M [L]
				//              Workspace: prefer M*M [VT] + M*N [L]
				//              At this point, L is resized as M by chunk.
				for i = 1; i <= (*n); i += chunk {
					blk = minint((*n)-i+1, chunk)
					err = goblas.Dgemm(NoTrans, NoTrans, *m, blk, *m, one, work.MatrixOff(ivt-1, *m, opts), *m, a.Off(0, i-1), *lda, zero, work.MatrixOff(il-1, ldwrkl, opts), ldwrkl)
					Dlacpy('F', m, &blk, work.MatrixOff(il-1, ldwrkl, opts), &ldwrkl, a.Off(0, i-1), lda)
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
				//              Workspace: need   M*M [L] + M [tau] + M    [work]
				//              Workspace: prefer M*M [L] + M [tau] + M*NB [work]
				Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Copy L to WORK(IL), zeroing out above it
				Dlacpy('L', m, m, a, lda, work.MatrixOff(il-1, ldwrkl, opts), &ldwrkl)
				Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, work.MatrixOff(il+ldwrkl-1, ldwrkl, opts), &ldwrkl)

				//              Generate Q in A
				//              Workspace: need   M*M [L] + M [tau] + M    [work]
				//              Workspace: prefer M*M [L] + M [tau] + M*NB [work]
				Dorglq(m, n, m, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				ie = itau
				itauq = ie + (*m)
				itaup = itauq + (*m)
				nwork = itaup + (*m)

				//              Bidiagonalize L in WORK(IU).
				//              Workspace: need   M*M [L] + 3*M [e, tauq, taup] + M      [work]
				//              Workspace: prefer M*M [L] + 3*M [e, tauq, taup] + 2*M*NB [work]
				Dgebrd(m, m, work.MatrixOff(il-1, ldwrkl, opts), &ldwrkl, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in U and computing right singular
				//              vectors of bidiagonal matrix in VT
				//              Workspace: need   M*M [L] + 3*M [e, tauq, taup] + BDSPAC
				Dbdsdc('U', 'I', m, s, work.Off(ie-1), u, ldu, vt, ldvt, dum, &idum, work.Off(nwork-1), iwork, info)

				//              Overwrite U by left singular vectors of L and VT
				//              by right singular vectors of L
				//              Workspace: need   M*M [L] + 3*M [e, tauq, taup] + M    [work]
				//              Workspace: prefer M*M [L] + 3*M [e, tauq, taup] + M*NB [work]
				Dormbr('Q', 'L', 'N', m, m, m, work.MatrixOff(il-1, ldwrkl, opts), &ldwrkl, work.Off(itauq-1), u, ldu, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				Dormbr('P', 'R', 'T', m, m, m, work.MatrixOff(il-1, ldwrkl, opts), &ldwrkl, work.Off(itaup-1), vt, ldvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Multiply right singular vectors of L in WORK(IL) by
				//              Q in A, storing result in VT
				//              Workspace: need   M*M [L]
				Dlacpy('F', m, m, vt, ldvt, work.MatrixOff(il-1, ldwrkl, opts), &ldwrkl)
				err = goblas.Dgemm(NoTrans, NoTrans, *m, *n, *m, one, work.MatrixOff(il-1, ldwrkl, opts), ldwrkl, a, *lda, zero, vt, *ldvt)

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
				//              Workspace: need   M*M [VT] + M [tau] + M    [work]
				//              Workspace: prefer M*M [VT] + M [tau] + M*NB [work]
				Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				Dlacpy('U', m, n, a, lda, vt, ldvt)

				//              Generate Q in VT
				//              Workspace: need   M*M [VT] + M [tau] + N    [work]
				//              Workspace: prefer M*M [VT] + M [tau] + N*NB [work]
				Dorglq(n, n, m, vt, ldvt, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Produce L in A, zeroing out other entries
				Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, a.Off(0, 1), lda)
				ie = itau
				itauq = ie + (*m)
				itaup = itauq + (*m)
				nwork = itaup + (*m)

				//              Bidiagonalize L in A
				//              Workspace: need   M*M [VT] + 3*M [e, tauq, taup] + M      [work]
				//              Workspace: prefer M*M [VT] + 3*M [e, tauq, taup] + 2*M*NB [work]
				Dgebrd(m, m, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in U and computing right singular
				//              vectors of bidiagonal matrix in WORK(IVT)
				//              Workspace: need   M*M [VT] + 3*M [e, tauq, taup] + BDSPAC
				Dbdsdc('U', 'I', m, s, work.Off(ie-1), u, ldu, work.MatrixOff(ivt-1, ldwkvt, opts), &ldwkvt, dum, &idum, work.Off(nwork-1), iwork, info)

				//              Overwrite U by left singular vectors of L and WORK(IVT)
				//              by right singular vectors of L
				//              Workspace: need   M*M [VT] + 3*M [e, tauq, taup]+ M    [work]
				//              Workspace: prefer M*M [VT] + 3*M [e, tauq, taup]+ M*NB [work]
				Dormbr('Q', 'L', 'N', m, m, m, a, lda, work.Off(itauq-1), u, ldu, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				Dormbr('P', 'R', 'T', m, m, m, a, lda, work.Off(itaup-1), work.MatrixOff(ivt-1, ldwkvt, opts), &ldwkvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				//              Multiply right singular vectors of L in WORK(IVT) by
				//              Q in VT, storing result in A
				//              Workspace: need   M*M [VT]
				err = goblas.Dgemm(NoTrans, NoTrans, *m, *n, *m, one, work.MatrixOff(ivt-1, ldwkvt, opts), ldwkvt, vt, *ldvt, zero, a, *lda)

				//              Copy right singular vectors of A from A to VT
				Dlacpy('F', m, n, a, lda, vt, ldvt)

			}

		} else {
			//           N .LT. MNTHR
			//
			//           Path 5t (N > M, but not much larger)
			//           Reduce to bidiagonal form without LQ decomposition
			ie = 1
			itauq = ie + (*m)
			itaup = itauq + (*m)
			nwork = itaup + (*m)

			//           Bidiagonalize A
			//           Workspace: need   3*M [e, tauq, taup] + N        [work]
			//           Workspace: prefer 3*M [e, tauq, taup] + (M+N)*NB [work]
			Dgebrd(m, n, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
			if wntqn {
				//              Path 5tn (N > M, JOBZ='N')
				//              Perform bidiagonal SVD, only computing singular values
				//              Workspace: need   3*M [e, tauq, taup] + BDSPAC
				Dbdsdc('L', 'N', m, s, work.Off(ie-1), dum.Matrix(1, opts), toPtr(1), dum.Matrix(1, opts), toPtr(1), dum, &idum, work.Off(nwork-1), iwork, info)
			} else if wntqo {
				//              Path 5to (N > M, JOBZ='O')
				ldwkvt = (*m)
				ivt = nwork
				if (*lwork) >= (*m)*(*n)+3*(*m)+bdspac {
					//                 WORK( IVT ) is M by N
					Dlaset('F', m, n, &zero, &zero, work.MatrixOff(ivt-1, ldwkvt, opts), &ldwkvt)
					nwork = ivt + ldwkvt*(*n)
					//                 IL is unused; silence compile warnings
					il = -1
				} else {
					//                 WORK( IVT ) is M by M
					nwork = ivt + ldwkvt*(*m)
					il = nwork

					//                 WORK(IL) is M by CHUNK
					chunk = ((*lwork) - (*m)*(*m) - 3*(*m)) / (*m)
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in U and computing right singular
				//              vectors of bidiagonal matrix in WORK(IVT)
				//              Workspace: need   3*M [e, tauq, taup] + M*M [VT] + BDSPAC
				Dbdsdc('L', 'I', m, s, work.Off(ie-1), u, ldu, work.MatrixOff(ivt-1, ldwkvt, opts), &ldwkvt, dum, &idum, work.Off(nwork-1), iwork, info)

				//              Overwrite U by left singular vectors of A
				//              Workspace: need   3*M [e, tauq, taup] + M*M [VT] + M    [work]
				//              Workspace: prefer 3*M [e, tauq, taup] + M*M [VT] + M*NB [work]
				Dormbr('Q', 'L', 'N', m, m, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

				if (*lwork) >= (*m)*(*n)+3*(*m)+bdspac {
					//                 Path 5to-fast
					//                 Overwrite WORK(IVT) by left singular vectors of A
					//                 Workspace: need   3*M [e, tauq, taup] + M*N [VT] + M    [work]
					//                 Workspace: prefer 3*M [e, tauq, taup] + M*N [VT] + M*NB [work]
					Dormbr('P', 'R', 'T', m, n, m, a, lda, work.Off(itaup-1), work.MatrixOff(ivt-1, ldwkvt, opts), &ldwkvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

					//                 Copy right singular vectors of A from WORK(IVT) to A
					Dlacpy('F', m, n, work.MatrixOff(ivt-1, ldwkvt, opts), &ldwkvt, a, lda)
				} else {
					//                 Path 5to-slow
					//                 Generate P**T in A
					//                 Workspace: need   3*M [e, tauq, taup] + M*M [VT] + M    [work]
					//                 Workspace: prefer 3*M [e, tauq, taup] + M*M [VT] + M*NB [work]
					Dorgbr('P', m, n, m, a, lda, work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)

					//                 Multiply Q in A by right singular vectors of
					//                 bidiagonal matrix in WORK(IVT), storing result in
					//                 WORK(IL) and copying to A
					//                 Workspace: need   3*M [e, tauq, taup] + M*M [VT] + M*NB [L]
					//                 Workspace: prefer 3*M [e, tauq, taup] + M*M [VT] + M*N  [L]
					for i = 1; i <= (*n); i += chunk {
						blk = minint((*n)-i+1, chunk)
						err = goblas.Dgemm(NoTrans, NoTrans, *m, blk, *m, one, work.MatrixOff(ivt-1, ldwkvt, opts), ldwkvt, a.Off(0, i-1), *lda, zero, work.MatrixOff(il-1, *m, opts), *m)
						Dlacpy('F', m, &blk, work.MatrixOff(il-1, *m, opts), m, a.Off(0, i-1), lda)
					}
				}
			} else if wntqs {
				//              Path 5ts (N > M, JOBZ='S')
				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in U and computing right singular
				//              vectors of bidiagonal matrix in VT
				//              Workspace: need   3*M [e, tauq, taup] + BDSPAC
				Dlaset('F', m, n, &zero, &zero, vt, ldvt)
				Dbdsdc('L', 'I', m, s, work.Off(ie-1), u, ldu, vt, ldvt, dum, &idum, work.Off(nwork-1), iwork, info)

				//              Overwrite U by left singular vectors of A and VT
				//              by right singular vectors of A
				//              Workspace: need   3*M [e, tauq, taup] + M    [work]
				//              Workspace: prefer 3*M [e, tauq, taup] + M*NB [work]
				Dormbr('Q', 'L', 'N', m, m, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				Dormbr('P', 'R', 'T', m, n, m, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
			} else if wntqa {
				//              Path 5ta (N > M, JOBZ='A')
				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in U and computing right singular
				//              vectors of bidiagonal matrix in VT
				//              Workspace: need   3*M [e, tauq, taup] + BDSPAC
				Dlaset('F', n, n, &zero, &zero, vt, ldvt)
				Dbdsdc('L', 'I', m, s, work.Off(ie-1), u, ldu, vt, ldvt, dum, &idum, work.Off(nwork-1), iwork, info)

				//              Set the right corner of VT to identity matrix
				if (*n) > (*m) {
					Dlaset('F', toPtr((*n)-(*m)), toPtr((*n)-(*m)), &zero, &one, vt.Off((*m)+1-1, (*m)+1-1), ldvt)
				}

				//              Overwrite U by left singular vectors of A and VT
				//              by right singular vectors of A
				//              Workspace: need   3*M [e, tauq, taup] + N    [work]
				//              Workspace: prefer 3*M [e, tauq, taup] + N*NB [work]
				Dormbr('Q', 'L', 'N', m, m, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
				Dormbr('P', 'R', 'T', n, n, m, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(nwork-1), toPtr((*lwork)-nwork+1), &ierr)
			}

		}

	}

	//     Undo scaling if necessary
	if iscl == 1 {
		if anrm > bignum {
			Dlascl('G', toPtr(0), toPtr(0), &bignum, &anrm, &minmn, toPtr(1), s.Matrix(minmn, opts), &minmn, &ierr)
		}
		if anrm < smlnum {
			Dlascl('G', toPtr(0), toPtr(0), &smlnum, &anrm, &minmn, toPtr(1), s.Matrix(minmn, opts), &minmn, &ierr)
		}
	}

	//     Return optimal workspace in WORK(1)
	work.Set(0, float64(maxwrk))
}
