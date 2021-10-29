package golapack

import (
	"fmt"
	"math"

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
func Zgelss(m, n, nrhs int, a, b *mat.CMatrix, s *mat.Vector, rcond float64, work *mat.CVector, lwork int, rwork *mat.Vector) (rank, info int, err error) {
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
	minmn = min(m, n)
	maxmn = max(m, n)
	lquery = (lwork == -1)
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if b.Rows < max(1, maxmn) {
		err = fmt.Errorf("b.Rows < max(1, maxmn): b.Rows=%v, maxmn=%v", b.Rows, maxmn)
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       CWorkspace refers to complex workspace, and RWorkspace refers
	//       to real workspace. NB refers to the optimal block size for the
	//       immediately following subroutine, as returned by ILAENV.)
	if err == nil {
		minwrk = 1
		maxwrk = 1
		if minmn > 0 {
			mm = m
			mnthr = Ilaenv(6, "Zgelss", []byte{' '}, m, n, nrhs, -1)
			if m >= n && m >= mnthr {
				//              Path 1a - overdetermined, with many more rows than
				//                        columns
				//
				//              Compute space needed for Zgeqrf
				if err = Zgeqrf(m, n, a, dum, dum, -1); err != nil {
					panic(err)
				}
				// lworkZgeqrf = int(dum.GetRe(0))
				//              Compute space needed for Zunmqr
				if err = Zunmqr(Left, ConjTrans, m, nrhs, n, a, dum, b, dum, -1); err != nil {
					panic(err)
				}
				// lworkZunmqr = int(dum.GetRe(0))
				mm = n
				maxwrk = max(maxwrk, n+n*Ilaenv(1, "Zgeqrf", []byte{' '}, m, n, 1, 1))
				maxwrk = max(maxwrk, n+nrhs*Ilaenv(1, "Zunmqr", []byte("LC"), m, nrhs, n, 1))
			}
			if m >= n {
				//              Path 1 - overdetermined or exactly determined
				//
				//              Compute space needed for ZGEBRD
				if err = Zgebrd(mm, n, a, s, s, dum, dum, dum, -1); err != nil {
					panic(err)
				}
				lworkZgebrd = int(dum.GetRe(0))
				//              Compute space needed for ZUNMBR
				if err = Zunmbr('Q', Left, ConjTrans, mm, nrhs, n, a, dum, b, dum, -1); err != nil {
					panic(err)
				}
				lworkZunmbr = int(dum.GetRe(0))
				//              Compute space needed for ZUNGBR
				if err = Zungbr('P', n, n, n, a, dum, dum, -1); err != nil {
					panic(err)
				}
				lworkZungbr = int(dum.GetRe(0))
				//              Compute total workspace needed
				maxwrk = max(maxwrk, 2*n+lworkZgebrd)
				maxwrk = max(maxwrk, 2*n+lworkZunmbr)
				maxwrk = max(maxwrk, 2*n+lworkZungbr)
				maxwrk = max(maxwrk, n*nrhs)
				minwrk = 2*n + max(nrhs, m)
			}
			if n > m {
				minwrk = 2*m + max(nrhs, n)
				if n >= mnthr {
					//                 Path 2a - underdetermined, with many more columns
					//                 than rows
					//
					//                 Compute space needed for ZGELQF
					if err = Zgelqf(m, n, a, dum, dum, -1); err != nil {
						panic(err)
					}
					lworkZgelqf = int(dum.GetRe(0))
					//                 Compute space needed for ZGEBRD
					if err = Zgebrd(m, m, a, s, s, dum, dum, dum, -1); err != nil {
						panic(err)
					}
					lworkZgebrd = int(dum.GetRe(0))
					//                 Compute space needed for ZUNMBR
					if err = Zunmbr('Q', Left, ConjTrans, m, nrhs, n, a, dum, b, dum, -1); err != nil {
						panic(err)
					}
					lworkZunmbr = int(dum.GetRe(0))
					//                 Compute space needed for ZUNGBR
					if err = Zungbr('P', m, m, m, a, dum, dum, -1); err != nil {
						panic(err)
					}
					lworkZungbr = int(dum.GetRe(0))
					//                 Compute space needed for ZUNMLQ
					if err = Zunmlq(Left, ConjTrans, n, nrhs, m, a, dum, b, dum, -1); err != nil {
						panic(err)
					}
					lworkZunmlq = int(dum.GetRe(0))
					//                 Compute total workspace needed
					maxwrk = m + lworkZgelqf
					maxwrk = max(maxwrk, 3*m+m*m+lworkZgebrd)
					maxwrk = max(maxwrk, 3*m+m*m+lworkZunmbr)
					maxwrk = max(maxwrk, 3*m+m*m+lworkZungbr)
					if nrhs > 1 {
						maxwrk = max(maxwrk, m*m+m+m*nrhs)
					} else {
						maxwrk = max(maxwrk, m*m+2*m)
					}
					maxwrk = max(maxwrk, m+lworkZunmlq)
				} else {
					//                 Path 2 - underdetermined
					//
					//                 Compute space needed for ZGEBRD
					if err = Zgebrd(m, n, a, s, s, dum, dum, dum, -1); err != nil {
						panic(err)
					}
					lworkZgebrd = int(dum.GetRe(0))
					//                 Compute space needed for ZUNMBR
					if err = Zunmbr('Q', Left, ConjTrans, m, nrhs, m, a, dum, b, dum, -1); err != nil {
						panic(err)
					}
					lworkZunmbr = int(dum.GetRe(0))
					//                 Compute space needed for ZUNGBR
					if err = Zungbr('P', m, n, m, a, dum, dum, -1); err != nil {
						panic(err)
					}
					lworkZungbr = int(dum.GetRe(0))
					maxwrk = 2*m + lworkZgebrd
					maxwrk = max(maxwrk, 2*m+lworkZunmbr)
					maxwrk = max(maxwrk, 2*m+lworkZungbr)
					maxwrk = max(maxwrk, n*nrhs)
				}
			}
			maxwrk = max(minwrk, maxwrk)
		}
		work.SetRe(0, float64(maxwrk))

		if lwork < minwrk && !lquery {
			err = fmt.Errorf("lwork < minwrk && !lquery: lwork=%v, minwrk=%v, lquery=%v", lwork, minwrk, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zgelss", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 {
		rank = 0
		return
	}

	//     Get machine parameters
	eps = Dlamch(Precision)
	sfmin = Dlamch(SafeMinimum)
	smlnum = sfmin / eps
	bignum = one / smlnum
	smlnum, bignum = Dlabad(smlnum, bignum)

	//     Scale A if max element outside range [SMLNUM,BIGNUM]
	anrm = Zlange('M', m, n, a, rwork)
	iascl = 0
	if anrm > zero && anrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		if err = Zlascl('G', 0, 0, anrm, smlnum, m, n, a); err != nil {
			panic(err)
		}
		iascl = 1
	} else if anrm > bignum {
		//        Scale matrix norm down to BIGNUM
		if err = Zlascl('G', 0, 0, anrm, bignum, m, n, a); err != nil {
			panic(err)
		}
		iascl = 2
	} else if anrm == zero {
		//        Matrix all zero. Return zero solution.
		Zlaset(Full, max(m, n), nrhs, czero, czero, b)
		Dlaset(Full, minmn, 1, zero, zero, s.Matrix(minmn, opts))
		rank = 0
		goto label70
	}

	//     Scale B if max element outside range [SMLNUM,BIGNUM]
	bnrm = Zlange('M', m, nrhs, b, rwork)
	ibscl = 0
	if bnrm > zero && bnrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		if err = Zlascl('G', 0, 0, bnrm, smlnum, m, nrhs, b); err != nil {
			panic(err)
		}
		ibscl = 1
	} else if bnrm > bignum {
		//        Scale matrix norm down to BIGNUM
		if err = Zlascl('G', 0, 0, bnrm, bignum, m, nrhs, b); err != nil {
			panic(err)
		}
		ibscl = 2
	}

	//     Overdetermined case
	if m >= n {
		//        Path 1 - overdetermined or exactly determined
		mm = m
		if m >= mnthr {
			//           Path 1a - overdetermined, with many more rows than columns
			mm = n
			itau = 1
			iwork = itau + n

			//           Compute A=Q*R
			//           (CWorkspace: need 2*N, prefer N+N*NB)
			//           (RWorkspace: none)
			if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
				panic(err)
			}

			//           Multiply B by transpose(Q)
			//           (CWorkspace: need N+NRHS, prefer N+NRHS*NB)
			//           (RWorkspace: none)
			if err = Zunmqr(Left, ConjTrans, m, nrhs, n, a, work.Off(itau-1), b, work.Off(iwork-1), lwork-iwork+1); err != nil {
				panic(err)
			}

			//           Zero out below R
			if n > 1 {
				Zlaset(Lower, n-1, n-1, czero, czero, a.Off(1, 0))
			}
		}
		//
		ie = 1
		itauq = 1
		itaup = itauq + n
		iwork = itaup + n

		//        Bidiagonalize R in A
		//        (CWorkspace: need 2*N+MM, prefer 2*N+(MM+N)*NB)
		//        (RWorkspace: need N)
		if err = Zgebrd(mm, n, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}

		//        Multiply B by transpose of left bidiagonalizing vectors of R
		//        (CWorkspace: need 2*N+NRHS, prefer 2*N+NRHS*NB)
		//        (RWorkspace: none)
		if err = Zunmbr('Q', Left, ConjTrans, mm, nrhs, n, a, work.Off(itauq-1), b, work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}

		//        Generate right bidiagonalizing vectors of R in A
		//        (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
		//        (RWorkspace: none)
		if err = Zungbr('P', n, n, n, a, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}
		irwork = ie + n

		//        Perform bidiagonal QR iteration
		//          multiply B by transpose of left singular vectors
		//          compute right singular vectors in A
		//        (CWorkspace: none)
		//        (RWorkspace: need BDSPAC)
		if info, err = Zbdsqr(Upper, n, n, 0, nrhs, s, rwork.Off(ie-1), a, dum.CMatrix(1, opts), b, rwork.Off(irwork-1)); err != nil {
			panic(err)
		}
		if (*&info) != 0 {
			goto label70
		}

		//        Multiply B by reciprocals of singular values
		thr = math.Max(rcond*s.Get(0), sfmin)
		if rcond < zero {
			thr = math.Max(eps*s.Get(0), sfmin)
		}
		rank = 0
		for i = 1; i <= n; i++ {
			if s.Get(i-1) > thr {
				Zdrscl(nrhs, s.Get(i-1), b.CVector(i-1, 0))
				rank = rank + 1
			} else {
				Zlaset(Full, 1, nrhs, czero, czero, b.Off(i-1, 0))
			}
		}

		//        Multiply B by right singular vectors
		//        (CWorkspace: need N, prefer N*NRHS)
		//        (RWorkspace: none)
		if lwork >= b.Rows*nrhs && nrhs > 1 {
			if err = goblas.Zgemm(ConjTrans, NoTrans, n, nrhs, n, cone, a, b, czero, work.CMatrix(b.Rows, opts)); err != nil {
				panic(err)
			}
			Zlacpy(Full, n, nrhs, work.CMatrix(b.Rows, opts), b)
		} else if nrhs > 1 {
			chunk = lwork / n
			for i = 1; i <= nrhs; i += chunk {
				bl = min(nrhs-i+1, chunk)
				if err = goblas.Zgemm(ConjTrans, NoTrans, n, bl, n, cone, a, b.Off(0, i-1), czero, work.CMatrix(n, opts)); err != nil {
					panic(err)
				}
				Zlacpy(Full, n, bl, work.CMatrix(n, opts), b.Off(0, i-1))
			}
		} else {
			if err = goblas.Zgemv(ConjTrans, n, n, cone, a, b.CVector(0, 0, 1), czero, work.Off(0, 1)); err != nil {
				panic(err)
			}
			goblas.Zcopy(n, work.Off(0, 1), b.CVector(0, 0, 1))
		}

	} else if n >= mnthr && lwork >= 3*m+m*m+max(m, nrhs, n-2*m) {
		//        Underdetermined case, M much less than N
		//
		//        Path 2a - underdetermined, with many more columns than rows
		//        and sufficient workspace for an efficient algorithm
		ldwork = m
		if lwork >= 3*m+m*a.Rows+max(m, nrhs, n-2*m) {
			ldwork = a.Rows
		}
		itau = 1
		iwork = m + 1

		//        Compute A=L*Q
		//        (CWorkspace: need 2*M, prefer M+M*NB)
		//        (RWorkspace: none)
		if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}
		il = iwork

		//        Copy L to WORK(IL), zeroing out above it
		Zlacpy(Lower, m, m, a, work.CMatrixOff(il-1, ldwork, opts))
		Zlaset(Upper, m-1, m-1, czero, czero, work.CMatrixOff(il+ldwork-1, ldwork, opts))
		ie = 1
		itauq = il + ldwork*m
		itaup = itauq + m
		iwork = itaup + m

		//        Bidiagonalize L in WORK(IL)
		//        (CWorkspace: need M*M+4*M, prefer M*M+3*M+2*M*NB)
		//        (RWorkspace: need M)
		if err = Zgebrd(m, m, work.CMatrixOff(il-1, ldwork, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}

		//        Multiply B by transpose of left bidiagonalizing vectors of L
		//        (CWorkspace: need M*M+3*M+NRHS, prefer M*M+3*M+NRHS*NB)
		//        (RWorkspace: none)
		if err = Zunmbr('Q', Left, ConjTrans, m, nrhs, m, work.CMatrixOff(il-1, ldwork, opts), work.Off(itauq-1), b, work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}

		//        Generate right bidiagonalizing vectors of R in WORK(IL)
		//        (CWorkspace: need M*M+4*M-1, prefer M*M+3*M+(M-1)*NB)
		//        (RWorkspace: none)
		if err = Zungbr('P', m, m, m, work.CMatrixOff(il-1, ldwork, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}
		irwork = ie + m

		//        Perform bidiagonal QR iteration, computing right singular
		//        vectors of L in WORK(IL) and multiplying B by transpose of
		//        left singular vectors
		//        (CWorkspace: need M*M)
		//        (RWorkspace: need BDSPAC)
		if info, err = Zbdsqr(Upper, m, m, 0, nrhs, s, rwork.Off(ie-1), work.CMatrixOff(il-1, ldwork, opts), a, b, rwork.Off(irwork-1)); err != nil {
			panic(err)
		}
		if (*&info) != 0 {
			goto label70
		}

		//        Multiply B by reciprocals of singular values
		thr = math.Max(rcond*s.Get(0), sfmin)
		if rcond < zero {
			thr = math.Max(eps*s.Get(0), sfmin)
		}
		rank = 0
		for i = 1; i <= m; i++ {
			if s.Get(i-1) > thr {
				Zdrscl(nrhs, s.Get(i-1), b.CVector(i-1, 0))
				rank = rank + 1
			} else {
				Zlaset(Full, 1, nrhs, czero, czero, b.Off(i-1, 0))
			}
		}
		iwork = il + m*ldwork

		//        Multiply B by right singular vectors of L in WORK(IL)
		//        (CWorkspace: need M*M+2*M, prefer M*M+M+M*NRHS)
		//        (RWorkspace: none)
		if lwork >= b.Rows*nrhs+iwork-1 && nrhs > 1 {
			if err = goblas.Zgemm(ConjTrans, NoTrans, m, nrhs, m, cone, work.CMatrixOff(il-1, ldwork, opts), b, czero, work.CMatrixOff(iwork-1, b.Rows, opts)); err != nil {
				panic(err)
			}
			Zlacpy(Full, m, nrhs, work.CMatrixOff(iwork-1, b.Rows, opts), b)
		} else if nrhs > 1 {
			chunk = (lwork - iwork + 1) / m
			for i = 1; i <= nrhs; i += chunk {
				bl = min(nrhs-i+1, chunk)
				if err = goblas.Zgemm(ConjTrans, NoTrans, m, bl, m, cone, work.CMatrixOff(il-1, ldwork, opts), b.Off(0, i-1), czero, work.CMatrixOff(iwork-1, m, opts)); err != nil {
					panic(err)
				}
				Zlacpy(Full, m, bl, work.CMatrixOff(iwork-1, m, opts), b.Off(0, i-1))
			}
		} else {
			if err = goblas.Zgemv(ConjTrans, m, m, cone, work.CMatrixOff(il-1, ldwork, opts), b.CVector(0, 0, 1), czero, work.Off(iwork-1, 1)); err != nil {
				panic(err)
			}
			goblas.Zcopy(m, work.Off(iwork-1, 1), b.CVector(0, 0, 1))
		}

		//        Zero out below first M rows of B
		Zlaset(Full, n-m, nrhs, czero, czero, b.Off(m, 0))
		iwork = itau + m

		//        Multiply transpose(Q) by B
		//        (CWorkspace: need M+NRHS, prefer M+NHRS*NB)
		//        (RWorkspace: none)
		if err = Zunmlq(Left, ConjTrans, n, nrhs, m, a, work.Off(itau-1), b, work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}

	} else {
		//        Path 2 - remaining underdetermined cases
		ie = 1
		itauq = 1
		itaup = itauq + m
		iwork = itaup + m

		//        Bidiagonalize A
		//        (CWorkspace: need 3*M, prefer 2*M+(M+N)*NB)
		//        (RWorkspace: need N)
		if err = Zgebrd(m, n, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}

		//        Multiply B by transpose of left bidiagonalizing vectors
		//        (CWorkspace: need 2*M+NRHS, prefer 2*M+NRHS*NB)
		//        (RWorkspace: none)
		if err = Zunmbr('Q', Left, ConjTrans, m, nrhs, n, a, work.Off(itauq-1), b, work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}

		//        Generate right bidiagonalizing vectors in A
		//        (CWorkspace: need 3*M, prefer 2*M+M*NB)
		//        (RWorkspace: none)
		if err = Zungbr('P', m, n, m, a, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}
		irwork = ie + m

		//        Perform bidiagonal QR iteration,
		//           computing right singular vectors of A in A and
		//           multiplying B by transpose of left singular vectors
		//        (CWorkspace: none)
		//        (RWorkspace: need BDSPAC)
		if info, err = Zbdsqr(Lower, m, n, 0, nrhs, s, rwork.Off(ie-1), a, dum.CMatrix(1, opts), b, rwork.Off(irwork-1)); err != nil {
			panic(err)
		}
		if (*&info) != 0 {
			goto label70
		}

		//        Multiply B by reciprocals of singular values
		thr = math.Max(rcond*s.Get(0), sfmin)
		if rcond < zero {
			thr = math.Max(eps*s.Get(0), sfmin)
		}
		rank = 0
		for i = 1; i <= m; i++ {
			if s.Get(i-1) > thr {
				Zdrscl(nrhs, s.Get(i-1), b.CVector(i-1, 0))
				rank = rank + 1
			} else {
				Zlaset(Full, 1, nrhs, czero, czero, b.Off(i-1, 0))
			}
		}

		//        Multiply B by right singular vectors of A
		//        (CWorkspace: need N, prefer N*NRHS)
		//        (RWorkspace: none)
		if lwork >= b.Rows*nrhs && nrhs > 1 {
			if err = goblas.Zgemm(ConjTrans, NoTrans, n, nrhs, m, cone, a, b, czero, work.CMatrix(b.Rows, opts)); err != nil {
				panic(err)
			}
			Zlacpy(Full, n, nrhs, work.CMatrix(b.Rows, opts), b)
		} else if nrhs > 1 {
			chunk = lwork / n
			for i = 1; i <= nrhs; i += chunk {
				bl = min(nrhs-i+1, chunk)
				if err = goblas.Zgemm(ConjTrans, NoTrans, n, bl, m, cone, a, b.Off(0, i-1), czero, work.CMatrix(n, opts)); err != nil {
					panic(err)
				}
				Zlacpy(Full, n, bl, work.CMatrix(n, opts), b.Off(0, i-1))
			}
		} else {
			if err = goblas.Zgemv(ConjTrans, m, n, cone, a, b.CVector(0, 0, 1), czero, work.Off(0, 1)); err != nil {
				panic(err)
			}
			goblas.Zcopy(n, work.Off(0, 1), b.CVector(0, 0, 1))
		}
	}

	//     Undo scaling
	if iascl == 1 {
		if err = Zlascl('G', 0, 0, anrm, smlnum, n, nrhs, b); err != nil {
			panic(err)
		}
		if err = Dlascl('G', 0, 0, smlnum, anrm, minmn, 1, s.Matrix(minmn, opts)); err != nil {
			panic(err)
		}
	} else if iascl == 2 {
		if err = Zlascl('G', 0, 0, anrm, bignum, n, nrhs, b); err != nil {
			panic(err)
		}
		if err = Dlascl('G', 0, 0, bignum, anrm, minmn, 1, s.Matrix(minmn, opts)); err != nil {
			panic(err)
		}
	}
	if ibscl == 1 {
		if err = Zlascl('G', 0, 0, smlnum, bnrm, n, nrhs, b); err != nil {
			panic(err)
		}
	} else if ibscl == 2 {
		if err = Zlascl('G', 0, 0, bignum, bnrm, n, nrhs, b); err != nil {
			panic(err)
		}
	}
label70:
	;
	work.SetRe(0, float64(maxwrk))

	return
}
