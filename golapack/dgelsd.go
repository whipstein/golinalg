package golapack

import (
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgelsd computes the minimum-norm solution to a real linear least
// squares problem:
//     minimize 2-norm(| b - A*x |)
// using the singular value decomposition (SVD) of A. A is an M-by-N
// matrix which may be rank-deficient.
//
// Several right hand side vectors b and solution vectors x can be
// handled in a single call; they are stored as the columns of the
// M-by-NRHS right hand side matrix B and the N-by-NRHS solution
// matrix X.
//
// The problem is solved in three steps:
// (1) Reduce the coefficient matrix A to bidiagonal form with
//     Householder transformations, reducing the original problem
//     into a "bidiagonal least squares problem" (BLS)
// (2) Solve the BLS using a divide and conquer approach.
// (3) Apply back all the Householder transformations to solve
//     the original least squares problem.
//
// The effective rank of A is determined by treating as zero those
// singular values which are less than RCOND times the largest singular
// value.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Dgelsd(m, n, nrhs *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, s *mat.Vector, rcond *float64, rank *int, work *mat.Vector, lwork *int, iwork *[]int, info *int) {
	var lquery bool
	var anrm, bignum, bnrm, eps, one, sfmin, smlnum, two, zero float64
	var iascl, ibscl, ie, il, itau, itaup, itauq, ldwork, liwork, maxmn, maxwrk, minmn, minwrk, mm, mnthr, nlvl, nwork, smlsiz, wlalsd int

	zero = 0.0
	one = 1.0
	two = 2.0

	//     Test the input arguments.
	(*info) = 0
	minmn = min(*m, *n)
	maxmn = max(*m, *n)
	mnthr = Ilaenv(func() *int { y := 6; return &y }(), []byte("DGELSD"), []byte{' '}, m, n, nrhs, toPtr(-1))
	lquery = ((*lwork) == -1)
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*lda) < max(1, *m) {
		(*info) = -5
	} else if (*ldb) < max(1, maxmn) {
		(*info) = -7
	}

	smlsiz = Ilaenv(func() *int { y := 9; return &y }(), []byte("DGELSD"), []byte{' '}, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }())

	//     Compute workspace.
	//     (Note: Comments in the code beginning "Workspace:" describe the
	//     minimal amount of workspace needed at that point in the code,
	//     as well as the preferred amount for good performance.
	//     NB refers to the optimal block size for the immediately
	//     following subroutine, as returned by ILAENV.)
	minwrk = 1
	liwork = 1
	minmn = max(1, minmn)
	nlvl = max(int(math.Log(float64(minmn)/float64(smlsiz+1))/math.Log(two))+1, 0)

	if (*info) == 0 {
		maxwrk = 0
		liwork = 3*minmn*nlvl + 11*minmn
		mm = (*m)
		if (*m) >= (*n) && (*m) >= mnthr {
			//           Path 1a - overdetermined, with many more rows than columns.
			mm = (*n)
			maxwrk = max(maxwrk, (*n)+(*n)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DGEQRF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1)))
			maxwrk = max(maxwrk, (*n)+(*nrhs)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMQR"), []byte("LT"), m, nrhs, n, toPtr(-1)))
		}
		if (*m) >= (*n) {
			//           Path 1 - overdetermined or exactly determined.
			maxwrk = max(maxwrk, 3*(*n)+(mm+(*n))*Ilaenv(func() *int { y := 1; return &y }(), []byte("DGEBRD"), []byte{' '}, &mm, n, toPtr(-1), toPtr(-1)))
			maxwrk = max(maxwrk, 3*(*n)+(*nrhs)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMBR"), []byte("QLT"), &mm, nrhs, n, toPtr(-1)))
			maxwrk = max(maxwrk, 3*(*n)+((*n)-1)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMBR"), []byte("PLN"), n, nrhs, n, toPtr(-1)))
			wlalsd = 9*(*n) + 2*(*n)*smlsiz + 8*(*n)*nlvl + (*n)*(*nrhs) + int(math.Pow(float64(smlsiz+1), 2))
			maxwrk = max(maxwrk, 3*(*n)+wlalsd)
			minwrk = max(3*(*n)+mm, 3*(*n)+(*nrhs), 3*(*n)+wlalsd)
		}
		if (*n) > (*m) {
			wlalsd = (9 * (*m)) + 2*(*m)*smlsiz + 8*(*m)*nlvl + (*m)*(*nrhs) + int(math.Pow(float64(smlsiz+1), 2))
			if (*n) >= mnthr {
				//              Path 2a - underdetermined, with many more columns
				//              than rows.
				maxwrk = (*m) + (*m)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DGELQF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
				maxwrk = max(maxwrk, (*m)*(*m)+4*(*m)+2*(*m)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DGEBRD"), []byte{' '}, m, m, toPtr(-1), toPtr(-1)))
				maxwrk = max(maxwrk, (*m)*(*m)+4*(*m)+(*nrhs)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMBR"), []byte("QLT"), m, nrhs, m, toPtr(-1)))
				maxwrk = max(maxwrk, (*m)*(*m)+4*(*m)+((*m)-1)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMBR"), []byte("PLN"), m, nrhs, m, toPtr(-1)))
				if (*nrhs) > 1 {
					maxwrk = max(maxwrk, (*m)*(*m)+(*m)+(*m)*(*nrhs))
				} else {
					maxwrk = max(maxwrk, (*m)*(*m)+2*(*m))
				}
				maxwrk = max(maxwrk, (*m)+(*nrhs)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMLQ"), []byte("LT"), n, nrhs, m, toPtr(-1)))
				maxwrk = max(maxwrk, (*m)*(*m)+4*(*m)+wlalsd)
				//!     XXX: Ensure the Path 2a case below is triggered.  The workspace

				//!     calculation should use queries for all routines eventually.

				maxwrk = max(maxwrk, 4*(*m)+(*m)*(*m)+max(*m, 2*(*m)-4, *nrhs, (*n)-3*(*m)))
			} else {
				//              Path 2 - remaining underdetermined cases.
				maxwrk = 3*(*m) + ((*n)+(*m))*Ilaenv(func() *int { y := 1; return &y }(), []byte("DGEBRD"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
				maxwrk = max(maxwrk, 3*(*m)+(*nrhs)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMBR"), []byte("QLT"), m, nrhs, n, toPtr(-1)))
				maxwrk = max(maxwrk, 3*(*m)+(*m)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMBR"), []byte("PLN"), n, nrhs, m, toPtr(-1)))
				maxwrk = max(maxwrk, 3*(*m)+wlalsd)
			}
			minwrk = max(3*(*m)+(*nrhs), 3*(*m)+(*m), 3*(*m)+wlalsd)
		}
		minwrk = min(minwrk, maxwrk)
		work.Set(0, float64(maxwrk))
		(*iwork)[0] = liwork
		if (*lwork) < minwrk && !lquery {
			(*info) = -12
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGELSD"), -(*info))
		return
	} else if lquery {
		goto label10
	}

	//     Quick return if possible.
	if (*m) == 0 || (*n) == 0 {
		(*rank) = 0
		return
	}

	//     Get machine parameters.
	eps = Dlamch(Precision)
	sfmin = Dlamch(SafeMinimum)
	smlnum = sfmin / eps
	bignum = one / smlnum
	Dlabad(&smlnum, &bignum)

	//     Scale A if max entry outside range [SMLNUM,BIGNUM].
	anrm = Dlange('M', m, n, a, lda, work)
	iascl = 0
	if anrm > zero && anrm < smlnum {
		//        Scale matrix norm up to SMLNUM.
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &smlnum, m, n, a, lda, info)
		iascl = 1
	} else if anrm > bignum {
		//        Scale matrix norm down to BIGNUM.
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &bignum, m, n, a, lda, info)
		iascl = 2
	} else if anrm == zero {
		//        Off all zero. Return zero solution.
		Dlaset('F', toPtr(max(*m, *n)), nrhs, &zero, &zero, b, ldb)
		Dlaset('F', &minmn, func() *int { y := 1; return &y }(), &zero, &zero, s.Matrix(1, opts), func() *int { y := 1; return &y }())
		(*rank) = 0
		goto label10
	}

	//     Scale B if max entry outside range [SMLNUM,BIGNUM].
	bnrm = Dlange('M', m, nrhs, b, ldb, work)
	ibscl = 0
	if bnrm > zero && bnrm < smlnum {
		//        Scale matrix norm up to SMLNUM.
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrm, &smlnum, m, nrhs, b, ldb, info)
		ibscl = 1
	} else if bnrm > bignum {
		//        Scale matrix norm down to BIGNUM.
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrm, &bignum, m, nrhs, b, ldb, info)
		ibscl = 2
	}

	//     If M < N make sure certain entries of B are zero.
	if (*m) < (*n) {
		Dlaset('F', toPtr((*n)-(*m)), nrhs, &zero, &zero, b.Off((*m), 0), ldb)
	}

	//     Overdetermined case.
	if (*m) >= (*n) {
		//        Path 1 - overdetermined or exactly determined.
		mm = (*m)
		if (*m) >= mnthr {
			//           Path 1a - overdetermined, with many more rows than columns.
			mm = (*n)
			itau = 1
			nwork = itau + (*n)

			//           Compute A=Q*R.
			//           (Workspace: need 2*N, prefer N+N*NB)
			Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

			//           Multiply B by transpose(Q).
			//           (Workspace: need N+NRHS, prefer N+NRHS*NB)
			Dormqr('L', 'T', m, nrhs, n, a, lda, work.Off(itau-1), b, ldb, work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

			//           Zero out below R.
			if (*n) > 1 {
				Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, a.Off(1, 0), lda)
			}
		}

		ie = 1
		itauq = ie + (*n)
		itaup = itauq + (*n)
		nwork = itaup + (*n)

		//        Bidiagonalize R in A.
		//        (Workspace: need 3*N+MM, prefer 3*N+(MM+N)*NB)
		Dgebrd(&mm, n, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

		//        Multiply B by transpose of left bidiagonalizing vectors of R.
		//        (Workspace: need 3*N+NRHS, prefer 3*N+NRHS*NB)
		Dormbr('Q', 'L', 'T', &mm, nrhs, n, a, lda, work.Off(itauq-1), b, ldb, work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

		//        Solve the bidiagonal least squares problem.
		Dlalsd('U', &smlsiz, n, nrhs, s, work.Off(ie-1), b, ldb, rcond, rank, work.Off(nwork-1), iwork, info)
		if (*info) != 0 {
			goto label10
		}

		//        Multiply B by right bidiagonalizing vectors of R.
		Dormbr('P', 'L', 'N', n, nrhs, n, a, lda, work.Off(itaup-1), b, ldb, work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

	} else if (*n) >= mnthr && (*lwork) >= 4*(*m)+(*m)*(*m)+max(*m, 2*(*m)-4, *nrhs, (*n)-3*(*m), wlalsd) {
		//        Path 2a - underdetermined, with many more columns than rows
		//        and sufficient workspace for an efficient algorithm.
		ldwork = (*m)
		if (*lwork) >= max(4*(*m)+(*m)*(*lda)+max(*m, 2*(*m)-4, *nrhs, (*n)-3*(*m)), (*m)*(*lda)+(*m)+(*m)*(*nrhs), 4*(*m)+(*m)*(*lda)+wlalsd) {
			ldwork = (*lda)
		}
		itau = 1
		nwork = (*m) + 1

		//        Compute A=L*Q.
		//        (Workspace: need 2*M, prefer M+M*NB)
		Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)
		il = nwork

		//        Copy L to WORK(IL), zeroing out above its diagonal.
		Dlacpy('L', m, m, a, lda, work.MatrixOff(il-1, ldwork, opts), &ldwork)
		Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, work.MatrixOff(il+ldwork-1, ldwork, opts), &ldwork)
		ie = il + ldwork*(*m)
		itauq = ie + (*m)
		itaup = itauq + (*m)
		nwork = itaup + (*m)

		//        Bidiagonalize L in WORK(IL).
		//        (Workspace: need M*M+5*M, prefer M*M+4*M+2*M*NB)
		Dgebrd(m, m, work.MatrixOff(il-1, ldwork, opts), &ldwork, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

		//        Multiply B by transpose of left bidiagonalizing vectors of L.
		//        (Workspace: need M*M+4*M+NRHS, prefer M*M+4*M+NRHS*NB)
		Dormbr('Q', 'L', 'T', m, nrhs, m, work.MatrixOff(il-1, ldwork, opts), &ldwork, work.Off(itauq-1), b, ldb, work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

		//        Solve the bidiagonal least squares problem.
		Dlalsd('U', &smlsiz, m, nrhs, s, work.Off(ie-1), b, ldb, rcond, rank, work.Off(nwork-1), iwork, info)
		if (*info) != 0 {
			goto label10
		}

		//        Multiply B by right bidiagonalizing vectors of L.
		Dormbr('P', 'L', 'N', m, nrhs, m, work.MatrixOff(il-1, ldwork, opts), &ldwork, work.Off(itaup-1), b, ldb, work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

		//        Zero out below first M rows of B.
		Dlaset('F', toPtr((*n)-(*m)), nrhs, &zero, &zero, b.Off((*m), 0), ldb)
		nwork = itau + (*m)

		//        Multiply transpose(Q) by B.
		//        (Workspace: need M+NRHS, prefer M+NRHS*NB)
		Dormlq('L', 'T', n, nrhs, m, a, lda, work.Off(itau-1), b, ldb, work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

	} else {
		//        Path 2 - remaining underdetermined cases.
		ie = 1
		itauq = ie + (*m)
		itaup = itauq + (*m)
		nwork = itaup + (*m)

		//        Bidiagonalize A.
		//        (Workspace: need 3*M+N, prefer 3*M+(M+N)*NB)
		Dgebrd(m, n, a, lda, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

		//        Multiply B by transpose of left bidiagonalizing vectors.
		//        (Workspace: need 3*M+NRHS, prefer 3*M+NRHS*NB)
		Dormbr('Q', 'L', 'T', m, nrhs, n, a, lda, work.Off(itauq-1), b, ldb, work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

		//        Solve the bidiagonal least squares problem.
		Dlalsd('L', &smlsiz, m, nrhs, s, work.Off(ie-1), b, ldb, rcond, rank, work.Off(nwork-1), iwork, info)
		if (*info) != 0 {
			goto label10
		}

		//        Multiply B by right bidiagonalizing vectors of A.
		Dormbr('P', 'L', 'N', n, nrhs, m, a, lda, work.Off(itaup-1), b, ldb, work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

	}

	//     Undo scaling.
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

label10:
	;
	work.Set(0, float64(maxwrk))
	(*iwork)[0] = liwork
}
