package golapack

import (
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgelsd computes the minimum-norm solution to a real linear least
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
func Zgelsd(m, n, nrhs *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, s *mat.Vector, rcond *float64, rank *int, work *mat.CVector, lwork *int, rwork *mat.Vector, iwork *[]int, info *int) {
	var lquery bool
	var czero complex128
	var anrm, bignum, bnrm, eps, one, sfmin, smlnum, two, zero float64
	var iascl, ibscl, ie, il, itau, itaup, itauq, ldwork, liwork, lrwork, maxmn, maxwrk, minmn, minwrk, mm, mnthr, nlvl, nrwork, nwork, smlsiz int

	zero = 0.0
	one = 1.0
	two = 2.0
	czero = (0.0 + 0.0*1i)

	//     Test the input arguments.
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

	//     Compute workspace.
	//     (Note: Comments in the code beginning "Workspace:" describe the
	//     minimal amount of workspace needed at that point in the code,
	//     as well as the preferred amount for good performance.
	//     NB refers to the optimal block size for the immediately
	//     following subroutine, as returned by ILAENV.)
	if (*info) == 0 {
		minwrk = 1
		maxwrk = 1
		liwork = 1
		lrwork = 1
		if minmn > 0 {
			smlsiz = Ilaenv(func() *int { y := 9; return &y }(), []byte("ZGELSD"), []byte{' '}, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }())
			mnthr = Ilaenv(func() *int { y := 6; return &y }(), []byte("ZGELSD"), []byte{' '}, m, n, nrhs, toPtr(-1))
			nlvl = maxint(int(math.Log(float64(minmn)/float64(smlsiz+1))/math.Log(two))+1, 0)
			liwork = 3*minmn*nlvl + 11*minmn
			mm = (*m)
			if (*m) >= (*n) && (*m) >= mnthr {
				//              Path 1a - overdetermined, with many more rows than
				//                        columns.
				mm = (*n)
				maxwrk = maxint(maxwrk, (*n)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEQRF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1)))
				maxwrk = maxint(maxwrk, (*nrhs)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMQR"), []byte("LC"), m, nrhs, n, toPtr(-1)))
			}
			if (*m) >= (*n) {
				//              Path 1 - overdetermined or exactly determined.
				lrwork = 10*(*n) + 2*(*n)*smlsiz + 8*(*n)*nlvl + 3*smlsiz*(*nrhs) + maxint(int(math.Pow(float64(smlsiz+1), 2)), (*n)*(1+(*nrhs))+2*(*nrhs))
				maxwrk = maxint(maxwrk, 2*(*n)+(mm+(*n))*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEBRD"), []byte{' '}, &mm, n, toPtr(-1), toPtr(-1)))
				maxwrk = maxint(maxwrk, 2*(*n)+(*nrhs)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMBR"), []byte("QLC"), &mm, nrhs, n, toPtr(-1)))
				maxwrk = maxint(maxwrk, 2*(*n)+((*n)-1)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMBR"), []byte("PLN"), n, nrhs, n, toPtr(-1)))
				maxwrk = maxint(maxwrk, 2*(*n)+(*n)*(*nrhs))
				minwrk = maxint(2*(*n)+mm, 2*(*n)+(*n)*(*nrhs))
			}
			if (*n) > (*m) {
				lrwork = 10*(*m) + 2*(*m)*smlsiz + 8*(*m)*nlvl + 3*smlsiz*(*nrhs) + maxint(int(math.Pow(float64(smlsiz+1), 2)), (*n)*(1+(*nrhs))+2*(*nrhs))
				if (*n) >= mnthr {
					//                 Path 2a - underdetermined, with many more columns
					//                           than rows.
					maxwrk = (*m) + (*m)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGELQF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
					maxwrk = maxint(maxwrk, (*m)*(*m)+4*(*m)+2*(*m)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEBRD"), []byte{' '}, m, m, toPtr(-1), toPtr(-1)))
					maxwrk = maxint(maxwrk, (*m)*(*m)+4*(*m)+(*nrhs)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMBR"), []byte("QLC"), m, nrhs, m, toPtr(-1)))
					maxwrk = maxint(maxwrk, (*m)*(*m)+4*(*m)+((*m)-1)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMLQ"), []byte("LC"), n, nrhs, m, toPtr(-1)))
					if (*nrhs) > 1 {
						maxwrk = maxint(maxwrk, (*m)*(*m)+(*m)+(*m)*(*nrhs))
					} else {
						maxwrk = maxint(maxwrk, (*m)*(*m)+2*(*m))
					}
					maxwrk = maxint(maxwrk, (*m)*(*m)+4*(*m)+(*m)*(*nrhs))
					//!     XXX: Ensure the Path 2a case below is triggered.  The workspace

					//!     calculation should use queries for all routines eventually.

					maxwrk = maxint(maxwrk, 4*(*m)+(*m)*(*m)+maxint(*m, 2*(*m)-4, *nrhs, (*n)-3*(*m)))
				} else {
					//                 Path 2 - underdetermined.
					maxwrk = 2*(*m) + ((*n)+(*m))*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEBRD"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
					maxwrk = maxint(maxwrk, 2*(*m)+(*nrhs)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMBR"), []byte("QLC"), m, nrhs, m, toPtr(-1)))
					maxwrk = maxint(maxwrk, 2*(*m)+(*m)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMBR"), []byte("PLN"), n, nrhs, m, toPtr(-1)))
					maxwrk = maxint(maxwrk, 2*(*m)+(*m)*(*nrhs))
				}
				minwrk = maxint(2*(*m)+(*n), 2*(*m)+(*m)*(*nrhs))
			}
		}
		minwrk = minint(minwrk, maxwrk)
		work.SetRe(0, float64(maxwrk))
		(*iwork)[0] = liwork
		rwork.Set(0, float64(lrwork))

		if (*lwork) < minwrk && !lquery {
			(*info) = -12
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGELSD"), -(*info))
		return
	} else if lquery {
		return
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

	//     Scale A if maxint entry outside range [SMLNUM,BIGNUM].
	anrm = Zlange('M', m, n, a, lda, rwork)
	iascl = 0
	if anrm > zero && anrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &smlnum, m, n, a, lda, info)
		iascl = 1
	} else if anrm > bignum {
		//        Scale matrix norm down to BIGNUM.
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &bignum, m, n, a, lda, info)
		iascl = 2
	} else if anrm == zero {
		//        Matrix all zero. Return zero solution.
		Zlaset('F', toPtr(maxint(*m, *n)), nrhs, &czero, &czero, b, ldb)
		Dlaset('F', &minmn, func() *int { y := 1; return &y }(), &zero, &zero, s.Matrix(1, opts), func() *int { y := 1; return &y }())
		(*rank) = 0
		goto label10
	}

	//     Scale B if maxint entry outside range [SMLNUM,BIGNUM].
	bnrm = Zlange('M', m, nrhs, b, ldb, rwork)
	ibscl = 0
	if bnrm > zero && bnrm < smlnum {
		//        Scale matrix norm up to SMLNUM.
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrm, &smlnum, m, nrhs, b, ldb, info)
		ibscl = 1
	} else if bnrm > bignum {
		//        Scale matrix norm down to BIGNUM.
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrm, &bignum, m, nrhs, b, ldb, info)
		ibscl = 2
	}

	//     If M < N make sure B(M+1:N,:) = 0
	if (*m) < (*n) {
		Zlaset('F', toPtr((*n)-(*m)), nrhs, &czero, &czero, b.Off((*m)+1-1, 0), ldb)
	}

	//     Overdetermined case.
	if (*m) >= (*n) {
		//        Path 1 - overdetermined or exactly determined.
		mm = (*m)
		if (*m) >= mnthr {
			//           Path 1a - overdetermined, with many more rows than columns
			mm = (*n)
			itau = 1
			nwork = itau + (*n)

			//           Compute A=Q*R.
			//           (RWorkspace: need N)
			//           (CWorkspace: need N, prefer N*NB)
			Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

			//           Multiply B by transpose(Q).
			//           (RWorkspace: need N)
			//           (CWorkspace: need NRHS, prefer NRHS*NB)
			Zunmqr('L', 'C', m, nrhs, n, a, lda, work.Off(itau-1), b, ldb, work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

			//           Zero out below R.
			if (*n) > 1 {
				Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, a.Off(1, 0), lda)
			}
		}

		itauq = 1
		itaup = itauq + (*n)
		nwork = itaup + (*n)
		ie = 1
		nrwork = ie + (*n)

		//        Bidiagonalize R in A.
		//        (RWorkspace: need N)
		//        (CWorkspace: need 2*N+MM, prefer 2*N+(MM+N)*NB)
		Zgebrd(&mm, n, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

		//        Multiply B by transpose of left bidiagonalizing vectors of R.
		//        (CWorkspace: need 2*N+NRHS, prefer 2*N+NRHS*NB)
		Zunmbr('Q', 'L', 'C', &mm, nrhs, n, a, lda, work.Off(itauq-1), b, ldb, work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

		//        Solve the bidiagonal least squares problem.
		Zlalsd('U', &smlsiz, n, nrhs, s, rwork.Off(ie-1), b, ldb, rcond, rank, work.Off(nwork-1), rwork.Off(nrwork-1), iwork, info)
		if (*info) != 0 {
			goto label10
		}

		//        Multiply B by right bidiagonalizing vectors of R.
		Zunmbr('P', 'L', 'N', n, nrhs, n, a, lda, work.Off(itaup-1), b, ldb, work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

	} else if (*n) >= mnthr && (*lwork) >= 4*(*m)+(*m)*(*m)+maxint(*m, 2*(*m)-4, *nrhs, (*n)-3*(*m)) {
		//        Path 2a - underdetermined, with many more columns than rows
		//        and sufficient workspace for an efficient algorithm.
		ldwork = (*m)
		if (*lwork) >= maxint(4*(*m)+(*m)*(*lda)+maxint(*m, 2*(*m)-4, *nrhs, (*n)-3*(*m)), (*m)*(*lda)+(*m)+(*m)*(*nrhs)) {
			ldwork = (*lda)
		}
		itau = 1
		nwork = (*m) + 1

		//        Compute A=L*Q.
		//        (CWorkspace: need 2*M, prefer M+M*NB)
		Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)
		il = nwork

		//        Copy L to WORK(IL), zeroing out above its diagonal.
		Zlacpy('L', m, m, a, lda, work.CMatrixOff(il-1, ldwork, opts), &ldwork)
		Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, work.CMatrixOff(il+ldwork-1, ldwork, opts), &ldwork)
		itauq = il + ldwork*(*m)
		itaup = itauq + (*m)
		nwork = itaup + (*m)
		ie = 1
		nrwork = ie + (*m)

		//        Bidiagonalize L in WORK(IL).
		//        (RWorkspace: need M)
		//        (CWorkspace: need M*M+4*M, prefer M*M+4*M+2*M*NB)
		Zgebrd(m, m, work.CMatrixOff(il-1, ldwork, opts), &ldwork, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

		//        Multiply B by transpose of left bidiagonalizing vectors of L.
		//        (CWorkspace: need M*M+4*M+NRHS, prefer M*M+4*M+NRHS*NB)
		Zunmbr('Q', 'L', 'C', m, nrhs, m, work.CMatrixOff(il-1, ldwork, opts), &ldwork, work.Off(itauq-1), b, ldb, work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

		//        Solve the bidiagonal least squares problem.
		Zlalsd('U', &smlsiz, m, nrhs, s, rwork.Off(ie-1), b, ldb, rcond, rank, work.Off(nwork-1), rwork.Off(nrwork-1), iwork, info)
		if (*info) != 0 {
			goto label10
		}

		//        Multiply B by right bidiagonalizing vectors of L.
		Zunmbr('P', 'L', 'N', m, nrhs, m, work.CMatrixOff(il-1, ldwork, opts), &ldwork, work.Off(itaup-1), b, ldb, work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

		//        Zero out below first M rows of B.
		Zlaset('F', toPtr((*n)-(*m)), nrhs, &czero, &czero, b.Off((*m)+1-1, 0), ldb)
		nwork = itau + (*m)

		//        Multiply transpose(Q) by B.
		//        (CWorkspace: need NRHS, prefer NRHS*NB)
		Zunmlq('L', 'C', n, nrhs, m, a, lda, work.Off(itau-1), b, ldb, work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

	} else {
		//        Path 2 - remaining underdetermined cases.
		itauq = 1
		itaup = itauq + (*m)
		nwork = itaup + (*m)
		ie = 1
		nrwork = ie + (*m)

		//        Bidiagonalize A.
		//        (RWorkspace: need M)
		//        (CWorkspace: need 2*M+N, prefer 2*M+(M+N)*NB)
		Zgebrd(m, n, a, lda, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

		//        Multiply B by transpose of left bidiagonalizing vectors.
		//        (CWorkspace: need 2*M+NRHS, prefer 2*M+NRHS*NB)
		Zunmbr('Q', 'L', 'C', m, nrhs, n, a, lda, work.Off(itauq-1), b, ldb, work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

		//        Solve the bidiagonal least squares problem.
		Zlalsd('L', &smlsiz, m, nrhs, s, rwork.Off(ie-1), b, ldb, rcond, rank, work.Off(nwork-1), rwork.Off(nrwork-1), iwork, info)
		if (*info) != 0 {
			goto label10
		}

		//        Multiply B by right bidiagonalizing vectors of A.
		Zunmbr('P', 'L', 'N', n, nrhs, m, a, lda, work.Off(itaup-1), b, ldb, work.Off(nwork-1), toPtr((*lwork)-nwork+1), info)

	}

	//     Undo scaling.
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

label10:
	;
	work.SetRe(0, float64(maxwrk))
	(*iwork)[0] = liwork
	rwork.Set(0, float64(lrwork))
}
