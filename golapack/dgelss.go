package golapack

import (
	"fmt"
	"math"

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
func Dgelss(m, n, nrhs int, a, b *mat.Matrix, s *mat.Vector, rcond float64, work *mat.Vector, lwork int) (rank, info int, err error) {
	var lquery bool
	var anrm, bignum, bnrm, eps, one, sfmin, smlnum, thr, zero float64
	var bdspac, bl, chunk, i, iascl, ibscl, ie, il, itau, itaup, itauq, iwork, ldwork, lworkDgebrd, lworkDgelqf, lworkDgeqrf, lworkDorgbr, lworkDormbr, lworkDormlq, lworkDormqr, maxmn, maxwrk, minmn, minwrk, mm, mnthr int

	dum := vf(1)
	zero = 0.0
	one = 1.0

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
		err = fmt.Errorf("b.Rows < max(1, maxmn): b.Rows=%v, m=%v, n=%v", b.Rows, m, n)
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       NB refers to the optimal block size for the immediately
	//       following subroutine, as returned by ILAENV.)
	if err == nil {
		minwrk = 1
		maxwrk = 1
		if minmn > 0 {
			mm = m
			mnthr = Ilaenv(6, "Dgelss", []byte{' '}, m, n, nrhs, -1)
			if m >= n && m >= mnthr {
				//              Path 1a - overdetermined, with many more rows than
				//                        columns
				//
				//              Compute space needed for DGEQRF
				if err = Dgeqrf(m, n, a, dum, dum, -1); err != nil {
					panic(err)
				}
				lworkDgeqrf = int(dum.Get(0))
				//              Compute space needed for DORMQR
				if err = Dormqr(Left, Trans, m, nrhs, n, a, dum, b, dum, -1); err != nil {
					panic(err)
				}
				lworkDormqr = int(dum.Get(0))
				mm = n
				maxwrk = max(maxwrk, n+lworkDgeqrf)
				maxwrk = max(maxwrk, n+lworkDormqr)
			}
			if m >= n {
				//              Path 1 - overdetermined or exactly determined
				//
				//              Compute workspace needed for DBDSQR
				//
				bdspac = max(1, 5*n)
				//              Compute space needed for DGEBRD
				if err = Dgebrd(mm, n, a, s, dum, dum, dum, dum, -1); err != nil {
					panic(err)
				}
				lworkDgebrd = int(dum.Get(0))
				//              Compute space needed for DORMBR
				if err = Dormbr('Q', Left, Trans, mm, nrhs, n, a, dum, b, dum, -1); err != nil {
					panic(err)
				}
				lworkDormbr = int(dum.Get(0))
				//              Compute space needed for DORGBR
				if err = Dorgbr('P', n, n, n, a, dum, dum, -1); err != nil {
					panic(err)
				}
				lworkDorgbr = int(dum.Get(0))
				//              Compute total workspace needed
				maxwrk = max(maxwrk, 3*n+lworkDgebrd)
				maxwrk = max(maxwrk, 3*n+lworkDormbr)
				maxwrk = max(maxwrk, 3*n+lworkDorgbr)
				maxwrk = max(maxwrk, bdspac)
				maxwrk = max(maxwrk, n*nrhs)
				minwrk = max(3*n+mm, 3*n+nrhs, bdspac)
				maxwrk = max(minwrk, maxwrk)
			}
			if n > m {
				//              Compute workspace needed for DBDSQR
				bdspac = max(1, 5*m)
				minwrk = max(3*m+nrhs, 3*m+n, bdspac)
				if n >= mnthr {
					//                 Path 2a - underdetermined, with many more columns
					//                 than rows
					//
					//                 Compute space needed for DGELQF
					if err = Dgelqf(m, n, a, dum, dum, -1); err != nil {
						panic(err)
					}
					lworkDgelqf = int(dum.Get(0))
					//                 Compute space needed for DGEBRD
					if err = Dgebrd(m, m, a, s, dum, dum, dum, dum, -1); err != nil {
						panic(err)
					}
					lworkDgebrd = int(dum.Get(0))
					//                 Compute space needed for DORMBR
					if err = Dormbr('Q', Left, Trans, m, nrhs, n, a, dum, b, dum, -1); err != nil {
						panic(err)
					}
					lworkDormbr = int(dum.Get(0))
					//                 Compute space needed for DORGBR
					if err = Dorgbr('P', m, m, m, a, dum, dum, -1); err != nil {
						panic(err)
					}
					lworkDorgbr = int(dum.Get(0))
					//                 Compute space needed for DORMLQ
					if err = Dormlq(Left, Trans, n, nrhs, m, a, dum, b, dum, -1); err != nil {
						panic(err)
					}
					lworkDormlq = int(dum.Get(0))
					//                 Compute total workspace needed
					maxwrk = m + lworkDgelqf
					maxwrk = max(maxwrk, m*m+4*m+lworkDgebrd)
					maxwrk = max(maxwrk, m*m+4*m+lworkDormbr)
					maxwrk = max(maxwrk, m*m+4*m+lworkDorgbr)
					maxwrk = max(maxwrk, m*m+m+bdspac)
					if nrhs > 1 {
						maxwrk = max(maxwrk, m*m+m+m*nrhs)
					} else {
						maxwrk = max(maxwrk, m*m+2*m)
					}
					maxwrk = max(maxwrk, m+lworkDormlq)
				} else {
					//                 Path 2 - underdetermined
					//
					//                 Compute space needed for DGEBRD
					if err = Dgebrd(m, n, a, s, dum, dum, dum, dum, -1); err != nil {
						panic(err)
					}
					lworkDgebrd = int(dum.Get(0))
					//                 Compute space needed for DORMBR
					if err = Dormbr('Q', Left, Trans, m, nrhs, m, a, dum, b, dum, -1); err != nil {
						panic(err)
					}
					lworkDormbr = int(dum.Get(0))
					//                 Compute space needed for DORGBR
					if err = Dorgbr('P', m, n, m, a, dum, dum, -1); err != nil {
						panic(err)
					}
					lworkDorgbr = int(dum.Get(0))
					maxwrk = 3*m + lworkDgebrd
					maxwrk = max(maxwrk, 3*m+lworkDormbr)
					maxwrk = max(maxwrk, 3*m+lworkDorgbr)
					maxwrk = max(maxwrk, bdspac)
					maxwrk = max(maxwrk, n*nrhs)
				}
			}
			maxwrk = max(minwrk, maxwrk)
		}
		work.Set(0, float64(maxwrk))

		if lwork < minwrk && !lquery {
			err = fmt.Errorf("lwork < minwrk && !lquery: lwork=%v, m=%v, n=%v, lquery=%v", lwork, m, n, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dgelss", err)
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
	anrm = Dlange('M', m, n, a, work)
	iascl = 0
	if anrm > zero && anrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		if err = Dlascl('G', 0, 0, anrm, smlnum, m, n, a); err != nil {
			panic(err)
		}
		iascl = 1
	} else if anrm > bignum {
		//        Scale matrix norm down to BIGNUM
		if err = Dlascl('G', 0, 0, anrm, bignum, m, n, a); err != nil {
			panic(err)
		}
		iascl = 2
	} else if anrm == zero {
		//        Off all zero. Return zero solution.
		Dlaset(Full, max(m, n), nrhs, zero, zero, b)
		Dlaset(Full, minmn, 1, zero, zero, s.Matrix(minmn, opts))
		rank = 0
		goto label70
	}

	//     Scale B if max element outside range [SMLNUM,BIGNUM]
	bnrm = Dlange('M', m, nrhs, b, work)
	ibscl = 0
	if bnrm > zero && bnrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		if err = Dlascl('G', 0, 0, bnrm, smlnum, m, nrhs, b); err != nil {
			panic(err)
		}
		ibscl = 1
	} else if bnrm > bignum {
		//        Scale matrix norm down to BIGNUM
		if err = Dlascl('G', 0, 0, bnrm, bignum, m, nrhs, b); err != nil {
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
			//           (Workspace: need 2*N, prefer N+N*NB)
			if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
				panic(err)
			}

			//           Multiply B by transpose(Q)
			//           (Workspace: need N+NRHS, prefer N+NRHS*NB)
			if err = Dormqr(Left, Trans, m, nrhs, n, a, work.Off(itau-1), b, work.Off(iwork-1), lwork-iwork+1); err != nil {
				panic(err)
			}

			//           Zero out below R
			if n > 1 {
				Dlaset(Lower, n-1, n-1, zero, zero, a.Off(1, 0))
			}
		}

		ie = 1
		itauq = ie + n
		itaup = itauq + n
		iwork = itaup + n

		//        Bidiagonalize R in A
		//        (Workspace: need 3*N+MM, prefer 3*N+(MM+N)*NB)
		if err = Dgebrd(mm, n, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}

		//        Multiply B by transpose of left bidiagonalizing vectors of R
		//        (Workspace: need 3*N+NRHS, prefer 3*N+NRHS*NB)
		if err = Dormbr('Q', Left, Trans, mm, nrhs, n, a, work.Off(itauq-1), b, work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}

		//        Generate right bidiagonalizing vectors of R in A
		//        (Workspace: need 4*N-1, prefer 3*N+(N-1)*NB)
		if err = Dorgbr('P', n, n, n, a, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}
		iwork = ie + n

		//        Perform bidiagonal QR iteration
		//          multiply B by transpose of left singular vectors
		//          compute right singular vectors in A
		//        (Workspace: need BDSPAC)
		if info, err = Dbdsqr(Upper, n, n, 0, nrhs, s, work.Off(ie-1), a, dum.Matrix(1, opts), b, work.Off(iwork-1)); info != 0 {
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
				Drscl(nrhs, s.Get(i-1), b.Off(i-1, 0).Vector(), b.Rows)
				rank = rank + 1
			} else {
				Dlaset(Full, 1, nrhs, zero, zero, b.Off(i-1, 0))
			}
		}

		//        Multiply B by right singular vectors
		//        (Workspace: need N, prefer N*NRHS)
		if lwork >= b.Rows*nrhs && nrhs > 1 {
			err = work.Matrix(b.Rows, opts).Gemm(Trans, NoTrans, n, nrhs, n, one, a, b, zero)
			Dlacpy(Full, n, nrhs, work.Matrix(b.Rows, opts), b)
		} else if nrhs > 1 {
			chunk = lwork / n
			for i = 1; i <= nrhs; i += chunk {
				bl = min(nrhs-i+1, chunk)
				err = work.Matrix(n, opts).Gemm(Trans, NoTrans, n, bl, n, one, a, b.Off(0, i-1), zero)
				Dlacpy(Full, n, bl, work.Matrix(n, opts), b.Off(0, i-1))
			}
		} else {
			err = work.Gemv(Trans, n, n, one, a, b.OffIdx(0).Vector(), 1, zero, 1)
			b.OffIdx(0).Vector().Copy(n, work, 1, 1)
		}

	} else if n >= mnthr && lwork >= 4*m+m*m+max(m, 2*m-4, nrhs, n-3*m) {
		//        Path 2a - underdetermined, with many more columns than rows
		//        and sufficient workspace for an efficient algorithm
		ldwork = m
		if lwork >= max(4*m+m*a.Rows+max(m, 2*m-4, nrhs, n-3*m), m*a.Rows+m+m*nrhs) {
			ldwork = a.Rows
		}
		itau = 1
		iwork = m + 1

		//        Compute A=L*Q
		//        (Workspace: need 2*M, prefer M+M*NB)
		if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}
		il = iwork

		//        Copy L to WORK(IL), zeroing out above it
		Dlacpy(Lower, m, m, a, work.Off(il-1).Matrix(ldwork, opts))
		Dlaset(Upper, m-1, m-1, zero, zero, work.Off(il+ldwork-1).Matrix(ldwork, opts))
		ie = il + ldwork*m
		itauq = ie + m
		itaup = itauq + m
		iwork = itaup + m

		//        Bidiagonalize L in WORK(IL)
		//        (Workspace: need M*M+5*M, prefer M*M+4*M+2*M*NB)
		if err = Dgebrd(m, m, work.Off(il-1).Matrix(ldwork, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}

		//        Multiply B by transpose of left bidiagonalizing vectors of L
		//        (Workspace: need M*M+4*M+NRHS, prefer M*M+4*M+NRHS*NB)
		if err = Dormbr('Q', Left, Trans, m, nrhs, m, work.Off(il-1).Matrix(ldwork, opts), work.Off(itauq-1), b, work.Off(iwork), lwork-iwork+1); err != nil {
			panic(err)
		}

		//        Generate right bidiagonalizing vectors of R in WORK(IL)
		//        (Workspace: need M*M+5*M-1, prefer M*M+4*M+(M-1)*NB)
		if err = Dorgbr('P', m, m, m, work.Off(il-1).Matrix(ldwork, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}
		iwork = ie + m

		//        Perform bidiagonal QR iteration,
		//           computing right singular vectors of L in WORK(IL) and
		//           multiplying B by transpose of left singular vectors
		//        (Workspace: need M*M+M+BDSPAC)
		if info, err = Dbdsqr(Upper, m, m, 0, nrhs, s, work.Off(ie-1), work.Off(il-1).Matrix(ldwork, opts), a, b, work.Off(iwork-1)); info != 0 {
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
				Drscl(nrhs, s.Get(i-1), b.Off(i-1, 0).Vector(), b.Rows)
				rank = rank + 1
			} else {
				Dlaset(Full, 1, nrhs, zero, zero, b.Off(i-1, 0))
			}
		}
		iwork = ie

		//        Multiply B by right singular vectors of L in WORK(IL)
		//        (Workspace: need M*M+2*M, prefer M*M+M+M*NRHS)
		if lwork >= b.Rows*nrhs+iwork-1 && nrhs > 1 {
			err = work.Off(iwork-1).Matrix(b.Rows, opts).Gemm(Trans, NoTrans, m, nrhs, m, one, work.Off(il-1).Matrix(ldwork, opts), b, zero)
			Dlacpy(Full, m, nrhs, work.Off(iwork-1).Matrix(b.Rows, opts), b)
		} else if nrhs > 1 {
			chunk = (lwork - iwork + 1) / m
			for i = 1; i <= nrhs; i += chunk {
				bl = min(nrhs-i+1, chunk)
				err = work.Off(iwork-1).Matrix(m, opts).Gemm(Trans, NoTrans, m, bl, m, one, work.Off(il-1).Matrix(ldwork, opts), b.Off(0, i-1), zero)
				Dlacpy(Full, m, bl, work.Off(iwork-1).Matrix(m, opts), b.Off(0, i-1))
			}
		} else {
			err = work.Off(iwork-1).Gemv(Trans, m, m, one, work.Off(il-1).Matrix(ldwork, opts), b.Off(0, 0).Vector(), 1, zero, 1)
			b.Off(0, 0).Vector().Copy(m, work.Off(iwork-1), 1, 1)
		}

		//        Zero out below first M rows of B
		Dlaset(Full, n-m, nrhs, zero, zero, b.Off(m, 0))
		iwork = itau + m

		//        Multiply transpose(Q) by B
		//        (Workspace: need M+NRHS, prefer M+NRHS*NB)
		if err = Dormlq(Left, Trans, n, nrhs, m, a, work.Off(itau-1), b, work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}

	} else {
		//        Path 2 - remaining underdetermined cases
		ie = 1
		itauq = ie + m
		itaup = itauq + m
		iwork = itaup + m

		//        Bidiagonalize A
		//        (Workspace: need 3*M+N, prefer 3*M+(M+N)*NB)
		if err = Dgebrd(m, n, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}

		//        Multiply B by transpose of left bidiagonalizing vectors
		//        (Workspace: need 3*M+NRHS, prefer 3*M+NRHS*NB)
		if err = Dormbr('Q', Left, Trans, m, nrhs, n, a, work.Off(itauq-1), b, work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}

		//        Generate right bidiagonalizing vectors in A
		//        (Workspace: need 4*M, prefer 3*M+M*NB)
		if err = Dorgbr('P', m, n, m, a, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
			panic(err)
		}
		iwork = ie + m

		//        Perform bidiagonal QR iteration,
		//           computing right singular vectors of A in A and
		//           multiplying B by transpose of left singular vectors
		//        (Workspace: need BDSPAC)
		if info, err = Dbdsqr(Lower, m, n, 0, nrhs, s, work.Off(ie-1), a, dum.Matrix(1, opts), b, work.Off(iwork-1)); info != 0 {
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
				Drscl(nrhs, s.Get(i-1), b.Off(i-1, 0).Vector(), b.Rows)
				rank = rank + 1
			} else {
				Dlaset(Full, 1, nrhs, zero, zero, b.Off(i-1, 0))
			}
		}

		//        Multiply B by right singular vectors of A
		//        (Workspace: need N, prefer N*NRHS)
		if lwork >= b.Rows*nrhs && nrhs > 1 {
			err = work.Matrix(b.Rows, opts).Gemm(Trans, NoTrans, n, nrhs, m, one, a, b, zero)
			Dlacpy(Full, n, nrhs, work.Matrix(b.Rows, opts), b)
		} else if nrhs > 1 {
			chunk = lwork / n
			for i = 1; i <= nrhs; i += chunk {
				bl = min(nrhs-i+1, chunk)
				err = work.Matrix(n, opts).Gemm(Trans, NoTrans, n, bl, m, one, a, b.Off(0, i-1), zero)
				Dlacpy(Full, n, bl, work.Matrix(n, opts), b.Off(0, i-1))
			}
		} else {
			err = work.Gemv(Trans, m, n, one, a, b.OffIdx(0).Vector(), 1, zero, 1)
			b.OffIdx(0).Vector().Copy(n, work, 1, 1)
		}
	}

	//     Undo scaling
	if iascl == 1 {
		if err = Dlascl('G', 0, 0, anrm, smlnum, n, nrhs, b); err != nil {
			panic(err)
		}
		if err = Dlascl('G', 0, 0, smlnum, anrm, minmn, 1, s.Matrix(minmn, opts)); err != nil {
			panic(err)
		}
	} else if iascl == 2 {
		if err = Dlascl('G', 0, 0, anrm, bignum, n, nrhs, b); err != nil {
			panic(err)
		}
		if err = Dlascl('G', 0, 0, bignum, anrm, minmn, 1, s.Matrix(minmn, opts)); err != nil {
			panic(err)
		}
	}
	if ibscl == 1 {
		if err = Dlascl('G', 0, 0, smlnum, bnrm, n, nrhs, b); err != nil {
			panic(err)
		}
	} else if ibscl == 2 {
		if err = Dlascl('G', 0, 0, bignum, bnrm, n, nrhs, b); err != nil {
			panic(err)
		}
	}

label70:
	;
	work.Set(0, float64(maxwrk))

	return
}
