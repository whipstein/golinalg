package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgelss computes the minimum norm solution to a complex linear
// least squares problem:
//
// Minimize 2-norm(| b - A*x |).
//
// using the singular value decomposition (SVD) of A. A is an M-by-N
// matrix which may be rank-deficient.
//
// Several right hand side vectors b and solution vectors x can be
// handled in a single call; they are stored as the columns of the
// M-by-NRHS right hand side matrix B and the N-by-NRHS solution matrix
// X.
//
// The effective rank of A is determined by treating as zero those
// singular values which are less than RCOND times the largest singular
// value.
func Zgelss(m, n, nrhs *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, s *mat.Vector, rcond *float64, rank *int, work *mat.CVector, lwork *int, rwork *mat.Vector, info *int) {
	var lquery bool
	var cone, czero complex128
	var anrm, bignum, bnrm, eps, one, sfmin, smlnum, thr, zero float64
	var bl, chunk, i, iascl, ibscl, ie, il, irwork, itau, itaup, itauq, iwork, ldwork, lworkZgebrd, lworkZgelqf, lworkZungbr, lworkZunmbr, lworkZunmlq, maxmn, maxwrk, minmn, minwrk, mm, mnthr int

	dum := cvf(1)

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Test the input arguments
	(*info) = 0
	minmn = minint(*m, *n)
	maxmn = maxint(*m, *n)
	lquery = ((*lwork) == -1)
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *m) {
		(*info) = -5
	} else if (*ldb) < maxint(1, maxmn) {
		(*info) = -7
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       CWorkspace refers to complex workspace, and RWorkspace refers
	//       to real workspace. NB refers to the optimal block size for the
	//       immediately following subroutine, as returned by ILAENV.)
	if (*info) == 0 {
		minwrk = 1
		maxwrk = 1
		if minmn > 0 {
			mm = (*m)
			mnthr = Ilaenv(func() *int { y := 6; return &y }(), []byte("ZGELSS"), []byte{' '}, m, n, nrhs, toPtr(-1))
			if (*m) >= (*n) && (*m) >= mnthr {
				//              Path 1a - overdetermined, with many more rows than
				//                        columns
				//
				//              Compute space needed for ZGEQRF
				Zgeqrf(m, n, a, lda, dum.Off(0), dum.Off(0), toPtr(-1), info)
				// lworkZgeqrf = int(dum.GetRe(0))
				//              Compute space needed for ZUNMQR
				Zunmqr('L', 'C', m, nrhs, n, a, lda, dum.Off(0), b, ldb, dum.Off(0), toPtr(-1), info)
				// lworkZunmqr = int(dum.GetRe(0))
				mm = (*n)
				maxwrk = maxint(maxwrk, (*n)+(*n)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEQRF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1)))
				maxwrk = maxint(maxwrk, (*n)+(*nrhs)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMQR"), []byte("LC"), m, nrhs, n, toPtr(-1)))
			}
			if (*m) >= (*n) {
				//              Path 1 - overdetermined or exactly determined
				//
				//              Compute space needed for ZGEBRD
				Zgebrd(&mm, n, a, lda, s, s, dum.Off(0), dum.Off(0), dum.Off(0), toPtr(-1), info)
				lworkZgebrd = int(dum.GetRe(0))
				//              Compute space needed for ZUNMBR
				Zunmbr('Q', 'L', 'C', &mm, nrhs, n, a, lda, dum.Off(0), b, ldb, dum.Off(0), toPtr(-1), info)
				lworkZunmbr = int(dum.GetRe(0))
				//              Compute space needed for ZUNGBR
				Zungbr('P', n, n, n, a, lda, dum.Off(0), dum.Off(0), toPtr(-1), info)
				lworkZungbr = int(dum.GetRe(0))
				//              Compute total workspace needed
				maxwrk = maxint(maxwrk, 2*(*n)+lworkZgebrd)
				maxwrk = maxint(maxwrk, 2*(*n)+lworkZunmbr)
				maxwrk = maxint(maxwrk, 2*(*n)+lworkZungbr)
				maxwrk = maxint(maxwrk, (*n)*(*nrhs))
				minwrk = 2*(*n) + maxint(*nrhs, *m)
			}
			if (*n) > (*m) {
				minwrk = 2*(*m) + maxint(*nrhs, *n)
				if (*n) >= mnthr {
					//                 Path 2a - underdetermined, with many more columns
					//                 than rows
					//
					//                 Compute space needed for ZGELQF
					Zgelqf(m, n, a, lda, dum.Off(0), dum.Off(0), toPtr(-1), info)
					lworkZgelqf = int(dum.GetRe(0))
					//                 Compute space needed for ZGEBRD
					Zgebrd(m, m, a, lda, s, s, dum.Off(0), dum.Off(0), dum.Off(0), toPtr(-1), info)
					lworkZgebrd = int(dum.GetRe(0))
					//                 Compute space needed for ZUNMBR
					Zunmbr('Q', 'L', 'C', m, nrhs, n, a, lda, dum.Off(0), b, ldb, dum.Off(0), toPtr(-1), info)
					lworkZunmbr = int(dum.GetRe(0))
					//                 Compute space needed for ZUNGBR
					Zungbr('P', m, m, m, a, lda, dum.Off(0), dum.Off(0), toPtr(-1), info)
					lworkZungbr = int(dum.GetRe(0))
					//                 Compute space needed for ZUNMLQ
					Zunmlq('L', 'C', n, nrhs, m, a, lda, dum.Off(0), b, ldb, dum.Off(0), toPtr(-1), info)
					lworkZunmlq = int(dum.GetRe(0))
					//                 Compute total workspace needed
					maxwrk = (*m) + lworkZgelqf
					maxwrk = maxint(maxwrk, 3*(*m)+(*m)*(*m)+lworkZgebrd)
					maxwrk = maxint(maxwrk, 3*(*m)+(*m)*(*m)+lworkZunmbr)
					maxwrk = maxint(maxwrk, 3*(*m)+(*m)*(*m)+lworkZungbr)
					if (*nrhs) > 1 {
						maxwrk = maxint(maxwrk, (*m)*(*m)+(*m)+(*m)*(*nrhs))
					} else {
						maxwrk = maxint(maxwrk, (*m)*(*m)+2*(*m))
					}
					maxwrk = maxint(maxwrk, (*m)+lworkZunmlq)
				} else {
					//                 Path 2 - underdetermined
					//
					//                 Compute space needed for ZGEBRD
					Zgebrd(m, n, a, lda, s, s, dum.Off(0), dum.Off(0), dum.Off(0), toPtr(-1), info)
					lworkZgebrd = int(dum.GetRe(0))
					//                 Compute space needed for ZUNMBR
					Zunmbr('Q', 'L', 'C', m, nrhs, m, a, lda, dum.Off(0), b, ldb, dum.Off(0), toPtr(-1), info)
					lworkZunmbr = int(dum.GetRe(0))
					//                 Compute space needed for ZUNGBR
					Zungbr('P', m, n, m, a, lda, dum.Off(0), dum.Off(0), toPtr(-1), info)
					lworkZungbr = int(dum.GetRe(0))
					maxwrk = 2*(*m) + lworkZgebrd
					maxwrk = maxint(maxwrk, 2*(*m)+lworkZunmbr)
					maxwrk = maxint(maxwrk, 2*(*m)+lworkZungbr)
					maxwrk = maxint(maxwrk, (*n)*(*nrhs))
				}
			}
			maxwrk = maxint(minwrk, maxwrk)
		}
		work.SetRe(0, float64(maxwrk))

		if (*lwork) < minwrk && !lquery {
			(*info) = -12
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGELSS"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		(*rank) = 0
		return
	}

	//     Get machine parameters
	eps = Dlamch(Precision)
	sfmin = Dlamch(SafeMinimum)
	smlnum = sfmin / eps
	bignum = one / smlnum
	Dlabad(&smlnum, &bignum)

	//     Scale A if maxint element outside range [SMLNUM,BIGNUM]
	anrm = Zlange('M', m, n, a, lda, rwork)
	iascl = 0
	if anrm > zero && anrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &smlnum, m, n, a, lda, info)
		iascl = 1
	} else if anrm > bignum {
		//        Scale matrix norm down to BIGNUM
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &bignum, m, n, a, lda, info)
		iascl = 2
	} else if anrm == zero {
		//        Matrix all zero. Return zero solution.
		Zlaset('F', toPtr(maxint(*m, *n)), nrhs, &czero, &czero, b, ldb)
		Dlaset('F', &minmn, func() *int { y := 1; return &y }(), &zero, &zero, s.Matrix(minmn, opts), &minmn)
		(*rank) = 0
		goto label70
	}

	//     Scale B if maxint element outside range [SMLNUM,BIGNUM]
	bnrm = Zlange('M', m, nrhs, b, ldb, rwork)
	ibscl = 0
	if bnrm > zero && bnrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrm, &smlnum, m, nrhs, b, ldb, info)
		ibscl = 1
	} else if bnrm > bignum {
		//        Scale matrix norm down to BIGNUM
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrm, &bignum, m, nrhs, b, ldb, info)
		ibscl = 2
	}

	//     Overdetermined case
	if (*m) >= (*n) {
		//        Path 1 - overdetermined or exactly determined
		mm = (*m)
		if (*m) >= mnthr {
			//           Path 1a - overdetermined, with many more rows than columns
			mm = (*n)
			itau = 1
			iwork = itau + (*n)

			//           Compute A=Q*R
			//           (CWorkspace: need 2*N, prefer N+N*NB)
			//           (RWorkspace: none)
			Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)

			//           Multiply B by transpose(Q)
			//           (CWorkspace: need N+NRHS, prefer N+NRHS*NB)
			//           (RWorkspace: none)
			Zunmqr('L', 'C', m, nrhs, n, a, lda, work.Off(itau-1), b, ldb, work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)

			//           Zero out below R
			if (*n) > 1 {
				Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, a.Off(1, 0), lda)
			}
		}
		//
		ie = 1
		itauq = 1
		itaup = itauq + (*n)
		iwork = itaup + (*n)

		//        Bidiagonalize R in A
		//        (CWorkspace: need 2*N+MM, prefer 2*N+(MM+N)*NB)
		//        (RWorkspace: need N)
		Zgebrd(&mm, n, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)

		//        Multiply B by transpose of left bidiagonalizing vectors of R
		//        (CWorkspace: need 2*N+NRHS, prefer 2*N+NRHS*NB)
		//        (RWorkspace: none)
		Zunmbr('Q', 'L', 'C', &mm, nrhs, n, a, lda, work.Off(itauq-1), b, ldb, work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)

		//        Generate right bidiagonalizing vectors of R in A
		//        (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
		//        (RWorkspace: none)
		Zungbr('P', n, n, n, a, lda, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)
		irwork = ie + (*n)

		//        Perform bidiagonal QR iteration
		//          multiply B by transpose of left singular vectors
		//          compute right singular vectors in A
		//        (CWorkspace: none)
		//        (RWorkspace: need BDSPAC)
		Zbdsqr('U', n, n, func() *int { y := 0; return &y }(), nrhs, s, rwork.Off(ie-1), a, lda, dum.CMatrix(1, opts), func() *int { y := 1; return &y }(), b, ldb, rwork.Off(irwork-1), info)
		if (*info) != 0 {
			goto label70
		}

		//        Multiply B by reciprocals of singular values
		thr = maxf64((*rcond)*s.Get(0), sfmin)
		if (*rcond) < zero {
			thr = maxf64(eps*s.Get(0), sfmin)
		}
		(*rank) = 0
		for i = 1; i <= (*n); i++ {
			if s.Get(i-1) > thr {
				Zdrscl(nrhs, s.GetPtr(i-1), b.CVector(i-1, 0), ldb)
				(*rank) = (*rank) + 1
			} else {
				Zlaset('F', func() *int { y := 1; return &y }(), nrhs, &czero, &czero, b.Off(i-1, 0), ldb)
			}
		}

		//        Multiply B by right singular vectors
		//        (CWorkspace: need N, prefer N*NRHS)
		//        (RWorkspace: none)
		if (*lwork) >= (*ldb)*(*nrhs) && (*nrhs) > 1 {
			goblas.Zgemm(ConjTrans, NoTrans, n, nrhs, n, &cone, a, lda, b, ldb, &czero, work.CMatrix(*ldb, opts), ldb)
			Zlacpy('G', n, nrhs, work.CMatrix(*ldb, opts), ldb, b, ldb)
		} else if (*nrhs) > 1 {
			chunk = (*lwork) / (*n)
			for i = 1; i <= (*nrhs); i += chunk {
				bl = minint((*nrhs)-i+1, chunk)
				goblas.Zgemm(ConjTrans, NoTrans, n, &bl, n, &cone, a, lda, b.Off(0, i-1), ldb, &czero, work.CMatrix(*n, opts), n)
				Zlacpy('G', n, &bl, work.CMatrix(*n, opts), n, b.Off(0, i-1), ldb)
			}
		} else {
			goblas.Zgemv(ConjTrans, n, n, &cone, a, lda, b.CVector(0, 0), func() *int { y := 1; return &y }(), &czero, work, func() *int { y := 1; return &y }())
			goblas.Zcopy(n, work, func() *int { y := 1; return &y }(), b.CVector(0, 0), func() *int { y := 1; return &y }())
		}

	} else if (*n) >= mnthr && (*lwork) >= 3*(*m)+(*m)*(*m)+maxint(*m, *nrhs, (*n)-2*(*m)) {
		//        Underdetermined case, M much less than N
		//
		//        Path 2a - underdetermined, with many more columns than rows
		//        and sufficient workspace for an efficient algorithm
		ldwork = (*m)
		if (*lwork) >= 3*(*m)+(*m)*(*lda)+maxint(*m, *nrhs, (*n)-2*(*m)) {
			ldwork = (*lda)
		}
		itau = 1
		iwork = (*m) + 1

		//        Compute A=L*Q
		//        (CWorkspace: need 2*M, prefer M+M*NB)
		//        (RWorkspace: none)
		Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)
		il = iwork

		//        Copy L to WORK(IL), zeroing out above it
		Zlacpy('L', m, m, a, lda, work.CMatrixOff(il-1, ldwork, opts), &ldwork)
		Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, work.CMatrixOff(il+ldwork-1, ldwork, opts), &ldwork)
		ie = 1
		itauq = il + ldwork*(*m)
		itaup = itauq + (*m)
		iwork = itaup + (*m)

		//        Bidiagonalize L in WORK(IL)
		//        (CWorkspace: need M*M+4*M, prefer M*M+3*M+2*M*NB)
		//        (RWorkspace: need M)
		Zgebrd(m, m, work.CMatrixOff(il-1, ldwork, opts), &ldwork, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)

		//        Multiply B by transpose of left bidiagonalizing vectors of L
		//        (CWorkspace: need M*M+3*M+NRHS, prefer M*M+3*M+NRHS*NB)
		//        (RWorkspace: none)
		Zunmbr('Q', 'L', 'C', m, nrhs, m, work.CMatrixOff(il-1, ldwork, opts), &ldwork, work.Off(itauq-1), b, ldb, work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)

		//        Generate right bidiagonalizing vectors of R in WORK(IL)
		//        (CWorkspace: need M*M+4*M-1, prefer M*M+3*M+(M-1)*NB)
		//        (RWorkspace: none)
		Zungbr('P', m, m, m, work.CMatrixOff(il-1, ldwork, opts), &ldwork, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)
		irwork = ie + (*m)

		//        Perform bidiagonal QR iteration, computing right singular
		//        vectors of L in WORK(IL) and multiplying B by transpose of
		//        left singular vectors
		//        (CWorkspace: need M*M)
		//        (RWorkspace: need BDSPAC)
		Zbdsqr('U', m, m, func() *int { y := 0; return &y }(), nrhs, s, rwork.Off(ie-1), work.CMatrixOff(il-1, ldwork, opts), &ldwork, a, lda, b, ldb, rwork.Off(irwork-1), info)
		if (*info) != 0 {
			goto label70
		}

		//        Multiply B by reciprocals of singular values
		thr = maxf64((*rcond)*s.Get(0), sfmin)
		if (*rcond) < zero {
			thr = maxf64(eps*s.Get(0), sfmin)
		}
		(*rank) = 0
		for i = 1; i <= (*m); i++ {
			if s.Get(i-1) > thr {
				Zdrscl(nrhs, s.GetPtr(i-1), b.CVector(i-1, 0), ldb)
				(*rank) = (*rank) + 1
			} else {
				Zlaset('F', func() *int { y := 1; return &y }(), nrhs, &czero, &czero, b.Off(i-1, 0), ldb)
			}
		}
		iwork = il + (*m)*ldwork

		//        Multiply B by right singular vectors of L in WORK(IL)
		//        (CWorkspace: need M*M+2*M, prefer M*M+M+M*NRHS)
		//        (RWorkspace: none)
		if (*lwork) >= (*ldb)*(*nrhs)+iwork-1 && (*nrhs) > 1 {
			goblas.Zgemm(ConjTrans, NoTrans, m, nrhs, m, &cone, work.CMatrixOff(il-1, ldwork, opts), &ldwork, b, ldb, &czero, work.CMatrixOff(iwork-1, *ldb, opts), ldb)
			Zlacpy('G', m, nrhs, work.CMatrixOff(iwork-1, *ldb, opts), ldb, b, ldb)
		} else if (*nrhs) > 1 {
			chunk = ((*lwork) - iwork + 1) / (*m)
			for i = 1; i <= (*nrhs); i += chunk {
				bl = minint((*nrhs)-i+1, chunk)
				goblas.Zgemm(ConjTrans, NoTrans, m, &bl, m, &cone, work.CMatrixOff(il-1, ldwork, opts), &ldwork, b.Off(0, i-1), ldb, &czero, work.CMatrixOff(iwork-1, *m, opts), m)
				Zlacpy('G', m, &bl, work.CMatrixOff(iwork-1, *m, opts), m, b.Off(0, i-1), ldb)
			}
		} else {
			goblas.Zgemv(ConjTrans, m, m, &cone, work.CMatrixOff(il-1, ldwork, opts), &ldwork, b.CVector(0, 0), func() *int { y := 1; return &y }(), &czero, work.Off(iwork-1), func() *int { y := 1; return &y }())
			goblas.Zcopy(m, work.Off(iwork-1), func() *int { y := 1; return &y }(), b.CVector(0, 0), func() *int { y := 1; return &y }())
		}

		//        Zero out below first M rows of B
		Zlaset('F', toPtr((*n)-(*m)), nrhs, &czero, &czero, b.Off((*m)+1-1, 0), ldb)
		iwork = itau + (*m)

		//        Multiply transpose(Q) by B
		//        (CWorkspace: need M+NRHS, prefer M+NHRS*NB)
		//        (RWorkspace: none)
		Zunmlq('L', 'C', n, nrhs, m, a, lda, work.Off(itau-1), b, ldb, work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)

	} else {
		//        Path 2 - remaining underdetermined cases
		ie = 1
		itauq = 1
		itaup = itauq + (*m)
		iwork = itaup + (*m)

		//        Bidiagonalize A
		//        (CWorkspace: need 3*M, prefer 2*M+(M+N)*NB)
		//        (RWorkspace: need N)
		Zgebrd(m, n, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)

		//        Multiply B by transpose of left bidiagonalizing vectors
		//        (CWorkspace: need 2*M+NRHS, prefer 2*M+NRHS*NB)
		//        (RWorkspace: none)
		Zunmbr('Q', 'L', 'C', m, nrhs, n, a, lda, work.Off(itauq-1), b, ldb, work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)

		//        Generate right bidiagonalizing vectors in A
		//        (CWorkspace: need 3*M, prefer 2*M+M*NB)
		//        (RWorkspace: none)
		Zungbr('P', m, n, m, a, lda, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)
		irwork = ie + (*m)

		//        Perform bidiagonal QR iteration,
		//           computing right singular vectors of A in A and
		//           multiplying B by transpose of left singular vectors
		//        (CWorkspace: none)
		//        (RWorkspace: need BDSPAC)
		Zbdsqr('L', m, n, func() *int { y := 0; return &y }(), nrhs, s, rwork.Off(ie-1), a, lda, dum.CMatrix(1, opts), func() *int { y := 1; return &y }(), b, ldb, rwork.Off(irwork-1), info)
		if (*info) != 0 {
			goto label70
		}

		//        Multiply B by reciprocals of singular values
		thr = maxf64((*rcond)*s.Get(0), sfmin)
		if (*rcond) < zero {
			thr = maxf64(eps*s.Get(0), sfmin)
		}
		(*rank) = 0
		for i = 1; i <= (*m); i++ {
			if s.Get(i-1) > thr {
				Zdrscl(nrhs, s.GetPtr(i-1), b.CVector(i-1, 0), ldb)
				(*rank) = (*rank) + 1
			} else {
				Zlaset('F', func() *int { y := 1; return &y }(), nrhs, &czero, &czero, b.Off(i-1, 0), ldb)
			}
		}

		//        Multiply B by right singular vectors of A
		//        (CWorkspace: need N, prefer N*NRHS)
		//        (RWorkspace: none)
		if (*lwork) >= (*ldb)*(*nrhs) && (*nrhs) > 1 {
			goblas.Zgemm(ConjTrans, NoTrans, n, nrhs, m, &cone, a, lda, b, ldb, &czero, work.CMatrix(*ldb, opts), ldb)
			Zlacpy('G', n, nrhs, work.CMatrix(*ldb, opts), ldb, b, ldb)
		} else if (*nrhs) > 1 {
			chunk = (*lwork) / (*n)
			for i = 1; i <= (*nrhs); i += chunk {
				bl = minint((*nrhs)-i+1, chunk)
				goblas.Zgemm(ConjTrans, NoTrans, n, &bl, m, &cone, a, lda, b.Off(0, i-1), ldb, &czero, work.CMatrix(*n, opts), n)
				Zlacpy('F', n, &bl, work.CMatrix(*n, opts), n, b.Off(0, i-1), ldb)
			}
		} else {
			goblas.Zgemv(ConjTrans, m, n, &cone, a, lda, b.CVector(0, 0), func() *int { y := 1; return &y }(), &czero, work, func() *int { y := 1; return &y }())
			goblas.Zcopy(n, work, func() *int { y := 1; return &y }(), b.CVector(0, 0), func() *int { y := 1; return &y }())
		}
	}

	//     Undo scaling
	if iascl == 1 {
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &smlnum, n, nrhs, b, ldb, info)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &smlnum, &anrm, &minmn, func() *int { y := 1; return &y }(), s.Matrix(minmn, opts), &minmn, info)
	} else if iascl == 2 {
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &bignum, n, nrhs, b, ldb, info)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bignum, &anrm, &minmn, func() *int { y := 1; return &y }(), s.Matrix(minmn, opts), &minmn, info)
	}
	if ibscl == 1 {
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &smlnum, &bnrm, n, nrhs, b, ldb, info)
	} else if ibscl == 2 {
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bignum, &bnrm, n, nrhs, b, ldb, info)
	}
label70:
	;
	work.SetRe(0, float64(maxwrk))
}
