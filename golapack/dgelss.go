package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgelss computes the minimum norm solution to a real linear least
// squares problem:
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
func Dgelss(m, n, nrhs *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, s *mat.Vector, rcond *float64, rank *int, work *mat.Vector, lwork, info *int) {
	var lquery bool
	var anrm, bignum, bnrm, eps, one, sfmin, smlnum, thr, zero float64
	var bdspac, bl, chunk, i, iascl, ibscl, ie, il, itau, itaup, itauq, iwork, ldwork, lworkDgebrd, lworkDgelqf, lworkDgeqrf, lworkDorgbr, lworkDormbr, lworkDormlq, lworkDormqr, maxmn, maxwrk, minmn, minwrk, mm, mnthr int

	dum := vf(1)
	zero = 0.0
	one = 1.0

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
	//       NB refers to the optimal block size for the immediately
	//       following subroutine, as returned by ILAENV.)
	if (*info) == 0 {
		minwrk = 1
		maxwrk = 1
		if minmn > 0 {
			mm = (*m)
			mnthr = Ilaenv(func() *int { y := 6; return &y }(), []byte("DGELSS"), []byte{' '}, m, n, nrhs, toPtr(-1))
			if (*m) >= (*n) && (*m) >= mnthr {
				//              Path 1a - overdetermined, with many more rows than
				//                        columns
				//
				//              Compute space needed for DGEQRF
				Dgeqrf(m, n, a, lda, dum, dum, toPtr(-1), info)
				lworkDgeqrf = int(dum.Get(0))
				//              Compute space needed for DORMQR
				Dormqr('L', 'T', m, nrhs, n, a, lda, dum, b, ldb, dum, toPtr(-1), info)
				lworkDormqr = int(dum.Get(0))
				mm = (*n)
				maxwrk = maxint(maxwrk, (*n)+lworkDgeqrf)
				maxwrk = maxint(maxwrk, (*n)+lworkDormqr)
			}
			if (*m) >= (*n) {
				//              Path 1 - overdetermined or exactly determined
				//
				//              Compute workspace needed for DBDSQR
				//
				bdspac = maxint(1, 5*(*n))
				//              Compute space needed for DGEBRD
				Dgebrd(&mm, n, a, lda, s, dum, dum, dum, dum, toPtr(-1), info)
				lworkDgebrd = int(dum.Get(0))
				//              Compute space needed for DORMBR
				Dormbr('Q', 'L', 'T', &mm, nrhs, n, a, lda, dum, b, ldb, dum, toPtr(-1), info)
				lworkDormbr = int(dum.Get(0))
				//              Compute space needed for DORGBR
				Dorgbr('P', n, n, n, a, lda, dum, dum, toPtr(-1), info)
				lworkDorgbr = int(dum.Get(0))
				//              Compute total workspace needed
				maxwrk = maxint(maxwrk, 3*(*n)+lworkDgebrd)
				maxwrk = maxint(maxwrk, 3*(*n)+lworkDormbr)
				maxwrk = maxint(maxwrk, 3*(*n)+lworkDorgbr)
				maxwrk = maxint(maxwrk, bdspac)
				maxwrk = maxint(maxwrk, (*n)*(*nrhs))
				minwrk = maxint(3*(*n)+mm, 3*(*n)+(*nrhs), bdspac)
				maxwrk = maxint(minwrk, maxwrk)
			}
			if (*n) > (*m) {
				//              Compute workspace needed for DBDSQR
				bdspac = maxint(1, 5*(*m))
				minwrk = maxint(3*(*m)+(*nrhs), 3*(*m)+(*n), bdspac)
				if (*n) >= mnthr {
					//                 Path 2a - underdetermined, with many more columns
					//                 than rows
					//
					//                 Compute space needed for DGELQF
					Dgelqf(m, n, a, lda, dum, dum, toPtr(-1), info)
					lworkDgelqf = int(dum.Get(0))
					//                 Compute space needed for DGEBRD
					Dgebrd(m, m, a, lda, s, dum, dum, dum, dum, toPtr(-1), info)
					lworkDgebrd = int(dum.Get(0))
					//                 Compute space needed for DORMBR
					Dormbr('Q', 'L', 'T', m, nrhs, n, a, lda, dum, b, ldb, dum, toPtr(-1), info)
					lworkDormbr = int(dum.Get(0))
					//                 Compute space needed for DORGBR
					Dorgbr('P', m, m, m, a, lda, dum, dum, toPtr(-1), info)
					lworkDorgbr = int(dum.Get(0))
					//                 Compute space needed for DORMLQ
					Dormlq('L', 'T', n, nrhs, m, a, lda, dum, b, ldb, dum, toPtr(-1), info)
					lworkDormlq = int(dum.Get(0))
					//                 Compute total workspace needed
					maxwrk = (*m) + lworkDgelqf
					maxwrk = maxint(maxwrk, (*m)*(*m)+4*(*m)+lworkDgebrd)
					maxwrk = maxint(maxwrk, (*m)*(*m)+4*(*m)+lworkDormbr)
					maxwrk = maxint(maxwrk, (*m)*(*m)+4*(*m)+lworkDorgbr)
					maxwrk = maxint(maxwrk, (*m)*(*m)+(*m)+bdspac)
					if (*nrhs) > 1 {
						maxwrk = maxint(maxwrk, (*m)*(*m)+(*m)+(*m)*(*nrhs))
					} else {
						maxwrk = maxint(maxwrk, (*m)*(*m)+2*(*m))
					}
					maxwrk = maxint(maxwrk, (*m)+lworkDormlq)
				} else {
					//                 Path 2 - underdetermined
					//
					//                 Compute space needed for DGEBRD
					Dgebrd(m, n, a, lda, s, dum, dum, dum, dum, toPtr(-1), info)
					lworkDgebrd = int(dum.Get(0))
					//                 Compute space needed for DORMBR
					Dormbr('Q', 'L', 'T', m, nrhs, m, a, lda, dum, b, ldb, dum, toPtr(-1), info)
					lworkDormbr = int(dum.Get(0))
					//                 Compute space needed for DORGBR
					Dorgbr('P', m, n, m, a, lda, dum, dum, toPtr(-1), info)
					lworkDorgbr = int(dum.Get(0))
					maxwrk = 3*(*m) + lworkDgebrd
					maxwrk = maxint(maxwrk, 3*(*m)+lworkDormbr)
					maxwrk = maxint(maxwrk, 3*(*m)+lworkDorgbr)
					maxwrk = maxint(maxwrk, bdspac)
					maxwrk = maxint(maxwrk, (*n)*(*nrhs))
				}
			}
			maxwrk = maxint(minwrk, maxwrk)
		}
		work.Set(0, float64(maxwrk))

		if (*lwork) < minwrk && !lquery {
			(*info) = -12
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGELSS"), -(*info))
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
	anrm = Dlange('M', m, n, a, lda, work)
	iascl = 0
	if anrm > zero && anrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &smlnum, m, n, a, lda, info)
		iascl = 1
	} else if anrm > bignum {
		//        Scale matrix norm down to BIGNUM
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &bignum, m, n, a, lda, info)
		iascl = 2
	} else if anrm == zero {
		//        Off all zero. Return zero solution.
		Dlaset('F', toPtr(maxint(*m, *n)), nrhs, &zero, &zero, b, ldb)
		Dlaset('F', &minmn, func() *int { y := 1; return &y }(), &zero, &zero, s.Matrix(minmn, opts), &minmn)
		(*rank) = 0
		goto label70
	}

	//     Scale B if maxint element outside range [SMLNUM,BIGNUM]
	bnrm = Dlange('M', m, nrhs, b, ldb, work)
	ibscl = 0
	if bnrm > zero && bnrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrm, &smlnum, m, nrhs, b, ldb, info)
		ibscl = 1
	} else if bnrm > bignum {
		//        Scale matrix norm down to BIGNUM
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrm, &bignum, m, nrhs, b, ldb, info)
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
			//           (Workspace: need 2*N, prefer N+N*NB)
			Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)

			//           Multiply B by transpose(Q)
			//           (Workspace: need N+NRHS, prefer N+NRHS*NB)
			Dormqr('L', 'T', m, nrhs, n, a, lda, work.Off(itau-1), b, ldb, work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)

			//           Zero out below R
			if (*n) > 1 {
				Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, a.Off(1, 0), lda)
			}
		}

		ie = 1
		itauq = ie + (*n)
		itaup = itauq + (*n)
		iwork = itaup + (*n)

		//        Bidiagonalize R in A
		//        (Workspace: need 3*N+MM, prefer 3*N+(MM+N)*NB)
		Dgebrd(&mm, n, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)

		//        Multiply B by transpose of left bidiagonalizing vectors of R
		//        (Workspace: need 3*N+NRHS, prefer 3*N+NRHS*NB)
		Dormbr('Q', 'L', 'T', &mm, nrhs, n, a, lda, work.Off(itauq-1), b, ldb, work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)

		//        Generate right bidiagonalizing vectors of R in A
		//        (Workspace: need 4*N-1, prefer 3*N+(N-1)*NB)
		Dorgbr('P', n, n, n, a, lda, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)
		iwork = ie + (*n)

		//        Perform bidiagonal QR iteration
		//          multiply B by transpose of left singular vectors
		//          compute right singular vectors in A
		//        (Workspace: need BDSPAC)
		Dbdsqr('U', n, n, func() *int { y := 0; return &y }(), nrhs, s, work.Off(ie-1), a, lda, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), b, ldb, work.Off(iwork-1), info)
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
				Drscl(nrhs, s.GetPtr(i-1), b.Vector(i-1, 0), ldb)
				(*rank) = (*rank) + 1
			} else {
				Dlaset('F', func() *int { y := 1; return &y }(), nrhs, &zero, &zero, b.Off(i-1, 0), ldb)
			}
		}

		//        Multiply B by right singular vectors
		//        (Workspace: need N, prefer N*NRHS)
		if (*lwork) >= (*ldb)*(*nrhs) && (*nrhs) > 1 {
			goblas.Dgemm(Trans, NoTrans, n, nrhs, n, &one, a, lda, b, ldb, &zero, work.Matrix(*ldb, opts), ldb)
			Dlacpy('G', n, nrhs, work.Matrix(*ldb, opts), ldb, b, ldb)
		} else if (*nrhs) > 1 {
			chunk = (*lwork) / (*n)
			for i = 1; i <= (*nrhs); i += chunk {
				bl = minint((*nrhs)-i+1, chunk)
				goblas.Dgemm(Trans, NoTrans, n, &bl, n, &one, a, lda, b.Off(0, i-1), ldb, &zero, work.Matrix(*n, opts), n)
				Dlacpy('G', n, &bl, work.Matrix(*n, opts), n, b.Off(0, i-1), ldb)
			}
		} else {
			goblas.Dgemv(Trans, n, n, &one, a, lda, b.VectorIdx(0), toPtr(1), &zero, work, toPtr(1))
			goblas.Dcopy(n, work, toPtr(1), b.VectorIdx(0), toPtr(1))
		}

	} else if (*n) >= mnthr && (*lwork) >= 4*(*m)+(*m)*(*m)+maxint(*m, 2*(*m)-4, *nrhs, (*n)-3*(*m)) {
		//        Path 2a - underdetermined, with many more columns than rows
		//        and sufficient workspace for an efficient algorithm
		ldwork = (*m)
		if (*lwork) >= maxint(4*(*m)+(*m)*(*lda)+maxint(*m, 2*(*m)-4, *nrhs, (*n)-3*(*m)), (*m)*(*lda)+(*m)+(*m)*(*nrhs)) {
			ldwork = (*lda)
		}
		itau = 1
		iwork = (*m) + 1

		//        Compute A=L*Q
		//        (Workspace: need 2*M, prefer M+M*NB)
		Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)
		il = iwork

		//        Copy L to WORK(IL), zeroing out above it
		Dlacpy('L', m, m, a, lda, work.MatrixOff(il-1, ldwork, opts), &ldwork)
		Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, work.MatrixOff(il+ldwork-1, ldwork, opts), &ldwork)
		ie = il + ldwork*(*m)
		itauq = ie + (*m)
		itaup = itauq + (*m)
		iwork = itaup + (*m)

		//        Bidiagonalize L in WORK(IL)
		//        (Workspace: need M*M+5*M, prefer M*M+4*M+2*M*NB)
		Dgebrd(m, m, work.MatrixOff(il-1, ldwork, opts), &ldwork, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)

		//        Multiply B by transpose of left bidiagonalizing vectors of L
		//        (Workspace: need M*M+4*M+NRHS, prefer M*M+4*M+NRHS*NB)
		Dormbr('Q', 'L', 'T', m, nrhs, m, work.MatrixOff(il-1, ldwork, opts), &ldwork, work.Off(itauq-1), b, ldb, work.Off(iwork), toPtr((*lwork)-iwork+1), info)

		//        Generate right bidiagonalizing vectors of R in WORK(IL)
		//        (Workspace: need M*M+5*M-1, prefer M*M+4*M+(M-1)*NB)
		Dorgbr('P', m, m, m, work.MatrixOff(il-1, ldwork, opts), &ldwork, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)
		iwork = ie + (*m)

		//        Perform bidiagonal QR iteration,
		//           computing right singular vectors of L in WORK(IL) and
		//           multiplying B by transpose of left singular vectors
		//        (Workspace: need M*M+M+BDSPAC)
		Dbdsqr('U', m, m, func() *int { y := 0; return &y }(), nrhs, s, work.Off(ie-1), work.MatrixOff(il-1, ldwork, opts), &ldwork, a, lda, b, ldb, work.Off(iwork-1), info)
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
				Drscl(nrhs, s.GetPtr(i-1), b.Vector(i-1, 0), ldb)
				(*rank) = (*rank) + 1
			} else {
				Dlaset('F', func() *int { y := 1; return &y }(), nrhs, &zero, &zero, b.Off(i-1, 0), ldb)
			}
		}
		iwork = ie

		//        Multiply B by right singular vectors of L in WORK(IL)
		//        (Workspace: need M*M+2*M, prefer M*M+M+M*NRHS)
		if (*lwork) >= (*ldb)*(*nrhs)+iwork-1 && (*nrhs) > 1 {
			goblas.Dgemm(Trans, NoTrans, m, nrhs, m, &one, work.MatrixOff(il-1, ldwork, opts), &ldwork, b, ldb, &zero, work.MatrixOff(iwork-1, *ldb, opts), ldb)
			Dlacpy('G', m, nrhs, work.MatrixOff(iwork-1, *ldb, opts), ldb, b, ldb)
		} else if (*nrhs) > 1 {
			chunk = ((*lwork) - iwork + 1) / (*m)
			for i = 1; i <= (*nrhs); i += chunk {
				bl = minint((*nrhs)-i+1, chunk)
				goblas.Dgemm(Trans, NoTrans, m, &bl, m, &one, work.MatrixOff(il-1, ldwork, opts), &ldwork, b.Off(0, i-1), ldb, &zero, work.MatrixOff(iwork-1, *m, opts), m)
				Dlacpy('G', m, &bl, work.MatrixOff(iwork-1, *m, opts), m, b.Off(0, i-1), ldb)
			}
		} else {
			goblas.Dgemv(Trans, m, m, &one, work.MatrixOff(il-1, ldwork, opts), &ldwork, b.Vector(0, 0), toPtr(1), &zero, work.Off(iwork-1), toPtr(1))
			goblas.Dcopy(m, work.Off(iwork-1), toPtr(1), b.Vector(0, 0), toPtr(1))
		}

		//        Zero out below first M rows of B
		Dlaset('F', toPtr((*n)-(*m)), nrhs, &zero, &zero, b.Off((*m)+1-1, 0), ldb)
		iwork = itau + (*m)

		//        Multiply transpose(Q) by B
		//        (Workspace: need M+NRHS, prefer M+NRHS*NB)
		Dormlq('L', 'T', n, nrhs, m, a, lda, work.Off(itau-1), b, ldb, work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)

	} else {
		//        Path 2 - remaining underdetermined cases
		ie = 1
		itauq = ie + (*m)
		itaup = itauq + (*m)
		iwork = itaup + (*m)

		//        Bidiagonalize A
		//        (Workspace: need 3*M+N, prefer 3*M+(M+N)*NB)
		Dgebrd(m, n, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)

		//        Multiply B by transpose of left bidiagonalizing vectors
		//        (Workspace: need 3*M+NRHS, prefer 3*M+NRHS*NB)
		Dormbr('Q', 'L', 'T', m, nrhs, n, a, lda, work.Off(itauq-1), b, ldb, work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)

		//        Generate right bidiagonalizing vectors in A
		//        (Workspace: need 4*M, prefer 3*M+M*NB)
		Dorgbr('P', m, n, m, a, lda, work.Off(itaup-1), work.Off(iwork-1), toPtr((*lwork)-iwork+1), info)
		iwork = ie + (*m)

		//        Perform bidiagonal QR iteration,
		//           computing right singular vectors of A in A and
		//           multiplying B by transpose of left singular vectors
		//        (Workspace: need BDSPAC)
		Dbdsqr('L', m, n, func() *int { y := 0; return &y }(), nrhs, s, work.Off(ie-1), a, lda, dum.Matrix(1, opts), func() *int { y := 1; return &y }(), b, ldb, work.Off(iwork-1), info)
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
				Drscl(nrhs, s.GetPtr(i-1), b.Vector(i-1, 0), ldb)
				(*rank) = (*rank) + 1
			} else {
				Dlaset('F', func() *int { y := 1; return &y }(), nrhs, &zero, &zero, b.Off(i-1, 0), ldb)
			}
		}

		//        Multiply B by right singular vectors of A
		//        (Workspace: need N, prefer N*NRHS)
		if (*lwork) >= (*ldb)*(*nrhs) && (*nrhs) > 1 {
			goblas.Dgemm(Trans, NoTrans, n, nrhs, m, &one, a, lda, b, ldb, &zero, work.Matrix(*ldb, opts), ldb)
			Dlacpy('F', n, nrhs, work.Matrix(*ldb, opts), ldb, b, ldb)
		} else if (*nrhs) > 1 {
			chunk = (*lwork) / (*n)
			for i = 1; i <= (*nrhs); i += chunk {
				bl = minint((*nrhs)-i+1, chunk)
				goblas.Dgemm(Trans, NoTrans, n, &bl, m, &one, a, lda, b.Off(0, i-1), ldb, &zero, work.Matrix(*n, opts), n)
				Dlacpy('F', n, &bl, work.Matrix(*n, opts), n, b.Off(0, i-1), ldb)
			}
		} else {
			goblas.Dgemv(Trans, m, n, &one, a, lda, b.VectorIdx(0), toPtr(1), &zero, work, toPtr(1))
			goblas.Dcopy(n, work, toPtr(1), b.VectorIdx(0), toPtr(1))
		}
	}

	//     Undo scaling
	if iascl == 1 {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &smlnum, n, nrhs, b, ldb, info)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &smlnum, &anrm, &minmn, func() *int { y := 1; return &y }(), s.Matrix(minmn, opts), &minmn, info)
	} else if iascl == 2 {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &bignum, n, nrhs, b, ldb, info)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bignum, &anrm, &minmn, func() *int { y := 1; return &y }(), s.Matrix(minmn, opts), &minmn, info)
	}
	if ibscl == 1 {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &smlnum, &bnrm, n, nrhs, b, ldb, info)
	} else if ibscl == 2 {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bignum, &bnrm, n, nrhs, b, ldb, info)
	}

label70:
	;
	work.Set(0, float64(maxwrk))
}
