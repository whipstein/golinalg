package golapack

import (
	"fmt"
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
// min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
// V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
// are the singular values of A; they are real and non-negative, and
// are returned in descending order.  The first min(m,n) columns of
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
func Dgesdd(jobz byte, m, n int, a *mat.Matrix, s *mat.Vector, u, vt *mat.Matrix, work *mat.Vector, lwork int, iwork *[]int) (info int, err error) {
	var lquery, wntqa, wntqas, wntqn, wntqo, wntqs bool
	var anrm, bignum, eps, one, smlnum, zero float64
	var bdspac, blk, chunk, i, ie, il, ir, iscl, itau, itaup, itauq, iu, ivt, ldwkvt, ldwrkl, ldwrkr, ldwrku, lworkDgebrdMm, lworkDgebrdMn, lworkDgebrdNn, lworkDgelqfMn, lworkDgeqrfMn, lworkDorglqMn, lworkDorglqNn, lworkDorgqrMm, lworkDorgqrMn, lworkDormbrPrtMm, lworkDormbrPrtMn, lworkDormbrPrtNn, lworkDormbrQlnMm, lworkDormbrQlnMn, lworkDormbrQlnNn, maxwrk, minmn, minwrk, mnthr, nwork, wrkbl int

	dum := vf(1)
	idum := make([]int, 1)

	zero = 0.0
	one = 1.0

	//     Test the input arguments
	minmn = min(m, n)
	wntqa = jobz == 'A'
	wntqs = jobz == 'S'
	wntqas = wntqa || wntqs
	wntqo = jobz == 'O'
	wntqn = jobz == 'N'
	lquery = (lwork == -1)

	if !(wntqa || wntqs || wntqo || wntqn) {
		err = fmt.Errorf("!(wntqa || wntqs || wntqo || wntqn): jobz='%c'", jobz)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if u.Rows < 1 || (wntqas && u.Rows < m) || (wntqo && m < n && u.Rows < m) {
		err = fmt.Errorf("u.Rows < 1 || (wntqas && u.Rows < m) || (wntqo && m < n && u.Rows < m): jobz='%c', u.Rows=%v, m=%v, n=%v", jobz, u.Rows, m, n)
	} else if vt.Rows < 1 || (wntqa && vt.Rows < n) || (wntqs && vt.Rows < minmn) || (wntqo && m >= n && vt.Rows < n) {
		err = fmt.Errorf("vt.Rows < 1 || (wntqa && vt.Rows < n) || (wntqs && vt.Rows < minmn) || (wntqo && m >= n && vt.Rows < n): jobz='%c', vt.Rows=%v, n=%v", jobz, vt.Rows, n)
	}

	//     Compute workspace
	//       Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace allocated at that point in the code,
	//       as well as the preferred amount for good performance.
	//       NB refers to the optimal block size for the immediately
	//       following subroutine, as returned by ILAENV.
	if err == nil {
		minwrk = 1
		maxwrk = 1
		bdspac = 0
		mnthr = int(minmn * 11.0 / 6.0)
		if m >= n && minmn > 0 {
			//           Compute space needed for DBDSDC
			if wntqn {
				//              dbdsdc needs only 4*N (or 6*N for uplo=L for LAPACK <= 3.6)
				//              keep 7*N for backwards compatibility.
				bdspac = 7 * n
			} else {
				bdspac = 3*n*n + 4*n
			}

			//           Compute space preferred for each routine
			if err = Dgebrd(m, n, dum.Matrix(m, opts), dum, dum, dum, dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDgebrdMn = int(dum.Get(0))

			if err = Dgebrd(n, n, dum.Matrix(m, opts), dum, dum, dum, dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDgebrdNn = int(dum.Get(0))

			if err = Dgeqrf(m, n, dum.Matrix(m, opts), dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDgeqrfMn = int(dum.Get(0))

			if err = Dorgbr('Q', n, n, n, dum.Matrix(m, opts), dum, dum, -1); err != nil {
				panic(err)
			}
			// lworkDorgbrQNn = int(dum.Get(0))

			if err = Dorgqr(m, m, n, dum.Matrix(m, opts), dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDorgqrMm = int(dum.Get(0))

			if err = Dorgqr(m, n, n, dum.Matrix(m, opts), dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDorgqrMn = int(dum.Get(0))

			if err = Dormbr('P', Right, Trans, n, n, n, dum.Matrix(n, opts), dum, dum.Matrix(n, opts), dum, -1); err != nil {
				panic(err)
			}
			lworkDormbrPrtNn = int(dum.Get(0))

			if err = Dormbr('Q', Left, NoTrans, n, n, n, dum.Matrix(n, opts), dum, dum.Matrix(n, opts), dum, -1); err != nil {
				panic(err)
			}
			lworkDormbrQlnNn = int(dum.Get(0))

			if err = Dormbr('Q', Left, NoTrans, m, n, n, dum.Matrix(m, opts), dum, dum.Matrix(m, opts), dum, -1); err != nil {
				panic(err)
			}
			lworkDormbrQlnMn = int(dum.Get(0))

			if err = Dormbr('Q', Left, NoTrans, m, m, n, dum.Matrix(m, opts), dum, dum.Matrix(m, opts), dum, -1); err != nil {
				panic(err)
			}
			lworkDormbrQlnMm = int(dum.Get(0))

			if m >= mnthr {
				if wntqn {
					//                 Path 1 (M >> N, JOBZ='N')
					wrkbl = n + lworkDgeqrfMn
					wrkbl = max(wrkbl, 3*n+lworkDgebrdNn)
					maxwrk = max(wrkbl, bdspac+n)
					minwrk = bdspac + n
				} else if wntqo {
					//                 Path 2 (M >> N, JOBZ='O')
					wrkbl = n + lworkDgeqrfMn
					wrkbl = max(wrkbl, n+lworkDorgqrMn)
					wrkbl = max(wrkbl, 3*n+lworkDgebrdNn)
					wrkbl = max(wrkbl, 3*n+lworkDormbrQlnNn)
					wrkbl = max(wrkbl, 3*n+lworkDormbrPrtNn)
					wrkbl = max(wrkbl, 3*n+bdspac)
					maxwrk = wrkbl + 2*n*n
					minwrk = bdspac + 2*n*n + 3*n
				} else if wntqs {
					//                 Path 3 (M >> N, JOBZ='S')
					wrkbl = n + lworkDgeqrfMn
					wrkbl = max(wrkbl, n+lworkDorgqrMn)
					wrkbl = max(wrkbl, 3*n+lworkDgebrdNn)
					wrkbl = max(wrkbl, 3*n+lworkDormbrQlnNn)
					wrkbl = max(wrkbl, 3*n+lworkDormbrPrtNn)
					wrkbl = max(wrkbl, 3*n+bdspac)
					maxwrk = wrkbl + n*n
					minwrk = bdspac + n*n + 3*n
				} else if wntqa {
					//                 Path 4 (M >> N, JOBZ='A')
					wrkbl = n + lworkDgeqrfMn
					wrkbl = max(wrkbl, n+lworkDorgqrMm)
					wrkbl = max(wrkbl, 3*n+lworkDgebrdNn)
					wrkbl = max(wrkbl, 3*n+lworkDormbrQlnNn)
					wrkbl = max(wrkbl, 3*n+lworkDormbrPrtNn)
					wrkbl = max(wrkbl, 3*n+bdspac)
					maxwrk = wrkbl + n*n
					minwrk = n*n + max(3*n+bdspac, n+m)
				}
			} else {
				//              Path 5 (M >= N, but not much larger)
				wrkbl = 3*n + lworkDgebrdMn
				if wntqn {
					//                 Path 5n (M >= N, jobz='N')
					maxwrk = max(wrkbl, 3*n+bdspac)
					minwrk = 3*n + max(m, bdspac)
				} else if wntqo {
					//                 Path 5o (M >= N, jobz='O')
					wrkbl = max(wrkbl, 3*n+lworkDormbrPrtNn)
					wrkbl = max(wrkbl, 3*n+lworkDormbrQlnMn)
					wrkbl = max(wrkbl, 3*n+bdspac)
					maxwrk = wrkbl + m*n
					minwrk = 3*n + max(m, n*n+bdspac)
				} else if wntqs {
					//                 Path 5s (M >= N, jobz='S')
					wrkbl = max(wrkbl, 3*n+lworkDormbrQlnMn)
					wrkbl = max(wrkbl, 3*n+lworkDormbrPrtNn)
					maxwrk = max(wrkbl, 3*n+bdspac)
					minwrk = 3*n + max(m, bdspac)
				} else if wntqa {
					//                 Path 5a (M >= N, jobz='A')
					wrkbl = max(wrkbl, 3*n+lworkDormbrQlnMm)
					wrkbl = max(wrkbl, 3*n+lworkDormbrPrtNn)
					maxwrk = max(wrkbl, 3*n+bdspac)
					minwrk = 3*n + max(m, bdspac)
				}
			}
		} else if minmn > 0 {
			//           Compute space needed for DBDSDC
			if wntqn {
				//              dbdsdc needs only 4*N (or 6*N for uplo=L for LAPACK <= 3.6)
				//              keep 7*N for backwards compatibility.
				bdspac = 7 * m
			} else {
				bdspac = 3*m*m + 4*m
			}

			//           Compute space preferred for each routine
			if err = Dgebrd(m, n, dum.Matrix(m, opts), dum, dum, dum, dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDgebrdMn = int(dum.Get(0))

			if err = Dgebrd(m, m, a, s, dum, dum, dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDgebrdMm = int(dum.Get(0))

			if err = Dgelqf(m, n, a, dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDgelqfMn = int(dum.Get(0))

			if err = Dorglq(n, n, m, dum.Matrix(n, opts), dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDorglqNn = int(dum.Get(0))

			if err = Dorglq(m, n, m, a.Off(0, 0).UpdateRows(m), dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDorglqMn = int(dum.Get(0))

			if err = Dorgbr('P', m, m, m, a, dum, dum, -1); err != nil {
				panic(err)
			}
			// lworkDorgbrPMm = int(dum.Get(0))

			if err = Dormbr('P', Right, Trans, m, m, m, dum.Matrix(m, opts), dum, dum.Matrix(m, opts), dum, -1); err != nil {
				panic(err)
			}
			lworkDormbrPrtMm = int(dum.Get(0))

			if err = Dormbr('P', Right, Trans, m, n, m, dum.Matrix(m, opts), dum, dum.Matrix(m, opts), dum, -1); err != nil {
				panic(err)
			}
			lworkDormbrPrtMn = int(dum.Get(0))

			if err = Dormbr('P', Right, Trans, n, n, m, dum.Matrix(n, opts), dum, dum.Matrix(n, opts), dum, -1); err != nil {
				panic(err)
			}
			lworkDormbrPrtNn = int(dum.Get(0))

			if err = Dormbr('Q', Left, NoTrans, m, m, m, dum.Matrix(m, opts), dum, dum.Matrix(m, opts), dum, -1); err != nil {
				panic(err)
			}
			lworkDormbrQlnMm = int(dum.Get(0))

			if n >= mnthr {
				if wntqn {
					//                 Path 1t (N >> M, JOBZ='N')
					wrkbl = m + lworkDgelqfMn
					wrkbl = max(wrkbl, 3*m+lworkDgebrdMm)
					maxwrk = max(wrkbl, bdspac+m)
					minwrk = bdspac + m
				} else if wntqo {
					//                 Path 2t (N >> M, JOBZ='O')
					wrkbl = m + lworkDgelqfMn
					wrkbl = max(wrkbl, m+lworkDorglqMn)
					wrkbl = max(wrkbl, 3*m+lworkDgebrdMm)
					wrkbl = max(wrkbl, 3*m+lworkDormbrQlnMm)
					wrkbl = max(wrkbl, 3*m+lworkDormbrPrtMm)
					wrkbl = max(wrkbl, 3*m+bdspac)
					maxwrk = wrkbl + 2*m*m
					minwrk = bdspac + 2*m*m + 3*m
				} else if wntqs {
					//                 Path 3t (N >> M, JOBZ='S')
					wrkbl = m + lworkDgelqfMn
					wrkbl = max(wrkbl, m+lworkDorglqMn)
					wrkbl = max(wrkbl, 3*m+lworkDgebrdMm)
					wrkbl = max(wrkbl, 3*m+lworkDormbrQlnMm)
					wrkbl = max(wrkbl, 3*m+lworkDormbrPrtMm)
					wrkbl = max(wrkbl, 3*m+bdspac)
					maxwrk = wrkbl + m*m
					minwrk = bdspac + m*m + 3*m
				} else if wntqa {
					//                 Path 4t (N >> M, JOBZ='A')
					wrkbl = m + lworkDgelqfMn
					wrkbl = max(wrkbl, m+lworkDorglqNn)
					wrkbl = max(wrkbl, 3*m+lworkDgebrdMm)
					wrkbl = max(wrkbl, 3*m+lworkDormbrQlnMm)
					wrkbl = max(wrkbl, 3*m+lworkDormbrPrtMm)
					wrkbl = max(wrkbl, 3*m+bdspac)
					maxwrk = wrkbl + m*m
					minwrk = m*m + max(3*m+bdspac, m+n)
				}
			} else {
				//              Path 5t (N > M, but not much larger)
				wrkbl = 3*m + lworkDgebrdMn
				if wntqn {
					//                 Path 5tn (N > M, jobz='N')
					maxwrk = max(wrkbl, 3*m+bdspac)
					minwrk = 3*m + max(n, bdspac)
				} else if wntqo {
					//                 Path 5to (N > M, jobz='O')
					wrkbl = max(wrkbl, 3*m+lworkDormbrQlnMm)
					wrkbl = max(wrkbl, 3*m+lworkDormbrPrtMn)
					wrkbl = max(wrkbl, 3*m+bdspac)
					maxwrk = wrkbl + m*n
					minwrk = 3*m + max(n, m*m+bdspac)
				} else if wntqs {
					//                 Path 5ts (N > M, jobz='S')
					wrkbl = max(wrkbl, 3*m+lworkDormbrQlnMm)
					wrkbl = max(wrkbl, 3*m+lworkDormbrPrtMn)
					maxwrk = max(wrkbl, 3*m+bdspac)
					minwrk = 3*m + max(n, bdspac)
				} else if wntqa {
					//                 Path 5ta (N > M, jobz='A')
					wrkbl = max(wrkbl, 3*m+lworkDormbrQlnMm)
					wrkbl = max(wrkbl, 3*m+lworkDormbrPrtNn)
					maxwrk = max(wrkbl, 3*m+bdspac)
					minwrk = 3*m + max(n, bdspac)
				}
			}
		}
		maxwrk = max(maxwrk, minwrk)
		work.Set(0, float64(maxwrk))

		if lwork < minwrk && !lquery {
			err = fmt.Errorf("lwork < minwrk && !lquery: lwork=%v, minwrk=%v, lquery=%v", lwork, minwrk, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dgesdd", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 {
		return
	}

	//     Get machine constants
	eps = Dlamch(Precision)
	smlnum = math.Sqrt(Dlamch(SafeMinimum)) / eps
	bignum = one / smlnum

	//     Scale A if max element outside range [SMLNUM,BIGNUM]
	anrm = Dlange('M', m, n, a, dum)
	iscl = 0
	if anrm > zero && anrm < smlnum {
		iscl = 1
		if err = Dlascl('G', 0, 0, anrm, smlnum, m, n, a); err != nil {
			panic(err)
		}
	} else if anrm > bignum {
		iscl = 1
		if err = Dlascl('G', 0, 0, anrm, bignum, m, n, a); err != nil {
			panic(err)
		}
	}

	if m >= n {
		//        A has at least as many rows as columns. If A has sufficiently
		//        more rows than columns, first reduce using the QR
		//        decomposition (if sufficient workspace available)
		if m >= mnthr {

			if wntqn {
				//              Path 1 (M >> N, JOBZ='N')
				//              No singular vectors to be computed
				itau = 1
				nwork = itau + n

				//              Compute A=Q*R
				//              Workspace: need   N [tau] + N    [work]
				//              Workspace: prefer N [tau] + N*NB [work]
				if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Zero out below R
				Dlaset(Lower, n-1, n-1, zero, zero, a.Off(1, 0))
				ie = 1
				itauq = ie + n
				itaup = itauq + n
				nwork = itaup + n

				//              Bidiagonalize R in A
				//              Workspace: need   3*N [e, tauq, taup] + N      [work]
				//              Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [work]
				if err = Dgebrd(n, n, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				nwork = ie + n

				//              Perform bidiagonal SVD, computing singular values only
				//              Workspace: need   N [e] + BDSPAC
				if err = Dbdsdc(Upper, 'N', n, s, work.Off(ie-1), dum.Matrix(1, opts), dum.Matrix(1, opts), dum, &idum, work.Off(nwork-1), iwork); err != nil {
					panic(err)
				}

			} else if wntqo {
				//              Path 2 (M >> N, JOBZ = 'O')
				//              N left singular vectors to be overwritten on A and
				//              N right singular vectors to be computed in VT
				ir = 1

				//              WORK(IR) is LDWRKR by N
				if lwork >= a.Rows*n+n*n+3*n+bdspac {
					ldwrkr = a.Rows
				} else {
					ldwrkr = (lwork - n*n - 3*n - bdspac) / n
				}
				itau = ir + ldwrkr*n
				nwork = itau + n

				//              Compute A=Q*R
				//              Workspace: need   N*N [R] + N [tau] + N    [work]
				//              Workspace: prefer N*N [R] + N [tau] + N*NB [work]
				if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy R to WORK(IR), zeroing out below it
				Dlacpy(Upper, n, n, a, work.MatrixOff(ir-1, ldwrkr, opts))
				Dlaset(Lower, n-1, n-1, zero, zero, work.MatrixOff(ir, ldwrkr, opts))

				//              Generate Q in A
				//              Workspace: need   N*N [R] + N [tau] + N    [work]
				//              Workspace: prefer N*N [R] + N [tau] + N*NB [work]
				if err = Dorgqr(m, n, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				ie = itau
				itauq = ie + n
				itaup = itauq + n
				nwork = itaup + n

				//              Bidiagonalize R in WORK(IR)
				//              Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N      [work]
				//              Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + 2*N*NB [work]
				if err = Dgebrd(n, n, work.MatrixOff(ir-1, ldwrkr, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              WORK(IU) is N by N
				iu = nwork
				nwork = iu + n*n

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in WORK(IU) and computing right
				//              singular vectors of bidiagonal matrix in VT
				//              Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N*N [U] + BDSPAC
				if err = Dbdsdc(Upper, 'I', n, s, work.Off(ie-1), work.MatrixOff(iu-1, n, opts), vt, dum, &idum, work.Off(nwork-1), iwork); err != nil {
					panic(err)
				}

				//              Overwrite WORK(IU) by left singular vectors of R
				//              and VT by right singular vectors of R
				//              Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N*N [U] + N    [work]
				//              Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*N [U] + N*NB [work]
				if err = Dormbr('Q', Left, NoTrans, n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), work.Off(itauq-1), work.MatrixOff(iu-1, n, opts), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				if err = Dormbr('P', Right, Trans, n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), work.Off(itaup-1), vt, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Multiply Q in A by left singular vectors of R in
				//              WORK(IU), storing result in WORK(IR) and copying to A
				//              Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N*N [U]
				//              Workspace: prefer M*N [R] + 3*N [e, tauq, taup] + N*N [U]
				for i = 1; i <= m; i += ldwrkr {
					chunk = min(m-i+1, ldwrkr)
					err = goblas.Dgemm(NoTrans, NoTrans, chunk, n, n, one, a.Off(i-1, 0), work.MatrixOff(iu-1, n, opts), zero, work.MatrixOff(ir-1, ldwrkr, opts))
					Dlacpy(Full, chunk, n, work.MatrixOff(ir-1, ldwrkr, opts), a.Off(i-1, 0))
				}

			} else if wntqs {
				//              Path 3 (M >> N, JOBZ='S')
				//              N left singular vectors to be computed in U and
				//              N right singular vectors to be computed in VT
				ir = 1

				//              WORK(IR) is N by N
				ldwrkr = n
				itau = ir + ldwrkr*n
				nwork = itau + n

				//              Compute A=Q*R
				//              Workspace: need   N*N [R] + N [tau] + N    [work]
				//              Workspace: prefer N*N [R] + N [tau] + N*NB [work]
				if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy R to WORK(IR), zeroing out below it
				Dlacpy(Upper, n, n, a, work.MatrixOff(ir-1, ldwrkr, opts))
				Dlaset(Lower, n-1, n-1, zero, zero, work.MatrixOff(ir, ldwrkr, opts))

				//              Generate Q in A
				//              Workspace: need   N*N [R] + N [tau] + N    [work]
				//              Workspace: prefer N*N [R] + N [tau] + N*NB [work]
				if err = Dorgqr(m, n, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				ie = itau
				itauq = ie + n
				itaup = itauq + n
				nwork = itaup + n

				//              Bidiagonalize R in WORK(IR)
				//              Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N      [work]
				//              Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + 2*N*NB [work]
				if err = Dgebrd(n, n, work.MatrixOff(ir-1, ldwrkr, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagoal matrix in U and computing right singular
				//              vectors of bidiagonal matrix in VT
				//              Workspace: need   N*N [R] + 3*N [e, tauq, taup] + BDSPAC
				if err = Dbdsdc(Upper, 'I', n, s, work.Off(ie-1), u, vt, dum, &idum, work.Off(nwork-1), iwork); err != nil {
					panic(err)
				}

				//              Overwrite U by left singular vectors of R and VT
				//              by right singular vectors of R
				//              Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N    [work]
				//              Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*NB [work]
				if err = Dormbr('Q', Left, NoTrans, n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), work.Off(itauq-1), u, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				if err = Dormbr('P', Right, Trans, n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), work.Off(itaup-1), vt, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Multiply Q in A by left singular vectors of R in
				//              WORK(IR), storing result in U
				//              Workspace: need   N*N [R]
				Dlacpy(Full, n, n, u, work.MatrixOff(ir-1, ldwrkr, opts))
				err = goblas.Dgemm(NoTrans, NoTrans, m, n, n, one, a, work.MatrixOff(ir-1, ldwrkr, opts), zero, u)

			} else if wntqa {
				//              Path 4 (M >> N, JOBZ='A')
				//              M left singular vectors to be computed in U and
				//              N right singular vectors to be computed in VT
				iu = 1

				//              WORK(IU) is N by N
				ldwrku = n
				itau = iu + ldwrku*n
				nwork = itau + n

				//              Compute A=Q*R, copying result to U
				//              Workspace: need   N*N [U] + N [tau] + N    [work]
				//              Workspace: prefer N*N [U] + N [tau] + N*NB [work]
				if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				Dlacpy(Lower, m, n, a, u)

				//              Generate Q in U
				//              Workspace: need   N*N [U] + N [tau] + M    [work]
				if err = Dorgqr(m, m, n, u, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Produce R in A, zeroing out other entries
				Dlaset(Lower, n-1, n-1, zero, zero, a.Off(1, 0))
				ie = itau
				itauq = ie + n
				itaup = itauq + n
				nwork = itaup + n

				//              Bidiagonalize R in A
				//              Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N      [work]
				//              Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + 2*N*NB [work]
				if err = Dgebrd(n, n, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in WORK(IU) and computing right
				//              singular vectors of bidiagonal matrix in VT
				//              Workspace: need   N*N [U] + 3*N [e, tauq, taup] + BDSPAC
				if err = Dbdsdc(Upper, 'I', n, s, work.Off(ie-1), work.MatrixOff(iu-1, n, opts), vt, dum, &idum, work.Off(nwork-1), iwork); err != nil {
					panic(err)
				}

				//              Overwrite WORK(IU) by left singular vectors of R and VT
				//              by right singular vectors of R
				//              Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N    [work]
				//              Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + N*NB [work]
				if err = Dormbr('Q', Left, NoTrans, n, n, n, a, work.Off(itauq-1), work.MatrixOff(iu-1, ldwrku, opts), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				if err = Dormbr('P', Right, Trans, n, n, n, a, work.Off(itaup-1), vt, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Multiply Q in U by left singular vectors of R in
				//              WORK(IU), storing result in A
				//              Workspace: need   N*N [U]
				err = goblas.Dgemm(NoTrans, NoTrans, m, n, n, one, u, work.MatrixOff(iu-1, ldwrku, opts), zero, a)

				//              Copy left singular vectors of A from A to U
				Dlacpy(Full, m, n, a, u)

			}

		} else {
			//           M .LT. MNTHR
			//
			//           Path 5 (M >= N, but not much larger)
			//           Reduce to bidiagonal form without QR decomposition

			ie = 1
			itauq = ie + n
			itaup = itauq + n
			nwork = itaup + n

			//           Bidiagonalize A
			//           Workspace: need   3*N [e, tauq, taup] + M        [work]
			//           Workspace: prefer 3*N [e, tauq, taup] + (M+N)*NB [work]
			if err = Dgebrd(m, n, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
				panic(err)
			}
			if wntqn {
				//              Path 5n (M >= N, JOBZ='N')
				//              Perform bidiagonal SVD, only computing singular values
				//              Workspace: need   3*N [e, tauq, taup] + BDSPAC
				if err = Dbdsdc(Upper, 'N', n, s, work.Off(ie-1), dum.Matrix(1, opts), dum.Matrix(1, opts), dum, &idum, work.Off(nwork-1), iwork); err != nil {
					panic(err)
				}
			} else if wntqo {
				//              Path 5o (M >= N, JOBZ='O')
				iu = nwork
				if lwork >= m*n+3*n+bdspac {
					//                 WORK( IU ) is M by N
					ldwrku = m
					nwork = iu + ldwrku*n
					Dlaset(Full, m, n, zero, zero, work.MatrixOff(iu-1, ldwrku, opts))
					//                 IR is unused; silence compile warnings
					ir = -1
				} else {
					//                 WORK( IU ) is N by N
					ldwrku = n
					nwork = iu + ldwrku*n

					//                 WORK(IR) is LDWRKR by N
					ir = nwork
					ldwrkr = (lwork - n*n - 3*n) / n
				}
				nwork = iu + ldwrku*n

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in WORK(IU) and computing right
				//              singular vectors of bidiagonal matrix in VT
				//              Workspace: need   3*N [e, tauq, taup] + N*N [U] + BDSPAC
				if err = Dbdsdc(Upper, 'I', n, s, work.Off(ie-1), work.MatrixOff(iu-1, ldwrku, opts), vt, dum, &idum, work.Off(nwork-1), iwork); err != nil {
					panic(err)
				}

				//              Overwrite VT by right singular vectors of A
				//              Workspace: need   3*N [e, tauq, taup] + N*N [U] + N    [work]
				//              Workspace: prefer 3*N [e, tauq, taup] + N*N [U] + N*NB [work]
				if err = Dormbr('P', Right, Trans, n, n, n, a, work.Off(itaup-1), vt, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				if lwork >= m*n+3*n+bdspac {
					//                 Path 5o-fast
					//                 Overwrite WORK(IU) by left singular vectors of A
					//                 Workspace: need   3*N [e, tauq, taup] + M*N [U] + N    [work]
					//                 Workspace: prefer 3*N [e, tauq, taup] + M*N [U] + N*NB [work]
					if err = Dormbr('Q', Left, NoTrans, m, n, n, a, work.Off(itauq-1), work.MatrixOff(iu-1, ldwrku, opts), work.Off(nwork-1), lwork-nwork+1); err != nil {
						panic(err)
					}

					//                 Copy left singular vectors of A from WORK(IU) to A
					Dlacpy(Full, m, n, work.MatrixOff(iu-1, ldwrku, opts), a)
				} else {
					//                 Path 5o-slow
					//                 Generate Q in A
					//                 Workspace: need   3*N [e, tauq, taup] + N*N [U] + N    [work]
					//                 Workspace: prefer 3*N [e, tauq, taup] + N*N [U] + N*NB [work]
					if err = Dorgbr('Q', m, n, n, a, work.Off(itauq-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
						panic(err)
					}

					//                 Multiply Q in A by left singular vectors of
					//                 bidiagonal matrix in WORK(IU), storing result in
					//                 WORK(IR) and copying to A
					//                 Workspace: need   3*N [e, tauq, taup] + N*N [U] + NB*N [R]
					//                 Workspace: prefer 3*N [e, tauq, taup] + N*N [U] + M*N  [R]
					for i = 1; i <= m; i += ldwrkr {
						chunk = min(m-i+1, ldwrkr)
						err = goblas.Dgemm(NoTrans, NoTrans, chunk, n, n, one, a.Off(i-1, 0), work.MatrixOff(iu-1, ldwrku, opts), zero, work.MatrixOff(ir-1, ldwrkr, opts))
						Dlacpy(Full, chunk, n, work.MatrixOff(ir-1, ldwrkr, opts), a.Off(i-1, 0))
					}
				}

			} else if wntqs {
				//              Path 5s (M >= N, JOBZ='S')
				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in U and computing right singular
				//              vectors of bidiagonal matrix in VT
				//              Workspace: need   3*N [e, tauq, taup] + BDSPAC
				Dlaset(Full, m, n, zero, zero, u)
				if err = Dbdsdc(Upper, 'I', n, s, work.Off(ie-1), u, vt, dum, &idum, work.Off(nwork-1), iwork); err != nil {
					panic(err)
				}

				//              Overwrite U by left singular vectors of A and VT
				//              by right singular vectors of A
				//              Workspace: need   3*N [e, tauq, taup] + N    [work]
				//              Workspace: prefer 3*N [e, tauq, taup] + N*NB [work]
				if err = Dormbr('Q', Left, NoTrans, m, n, n, a, work.Off(itauq-1), u, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				if err = Dormbr('P', Right, Trans, n, n, n, a, work.Off(itaup-1), vt, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
			} else if wntqa {
				//              Path 5a (M >= N, JOBZ='A')
				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in U and computing right singular
				//              vectors of bidiagonal matrix in VT
				//              Workspace: need   3*N [e, tauq, taup] + BDSPAC
				Dlaset(Full, m, m, zero, zero, u)
				if err = Dbdsdc(Upper, 'I', n, s, work.Off(ie-1), u, vt, dum, &idum, work.Off(nwork-1), iwork); err != nil {
					panic(err)
				}

				//              Set the right corner of U to identity matrix
				if m > n {
					Dlaset(Full, m-n, m-n, zero, one, u.Off(n, n))
				}

				//              Overwrite U by left singular vectors of A and VT
				//              by right singular vectors of A
				//              Workspace: need   3*N [e, tauq, taup] + M    [work]
				//              Workspace: prefer 3*N [e, tauq, taup] + M*NB [work]
				if err = Dormbr('Q', Left, NoTrans, m, m, n, a, work.Off(itauq-1), u, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				if err = Dormbr('P', Right, Trans, n, n, m, a, work.Off(itaup-1), vt, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
			}

		}

	} else {
		//        A has more columns than rows. If A has sufficiently more
		//        columns than rows, first reduce using the LQ decomposition (if
		//        sufficient workspace available)
		if n >= mnthr {

			if wntqn {
				//              Path 1t (N >> M, JOBZ='N')
				//              No singular vectors to be computed
				itau = 1
				nwork = itau + m

				//              Compute A=L*Q
				//              Workspace: need   M [tau] + M [work]
				//              Workspace: prefer M [tau] + M*NB [work]
				if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Zero out above L
				Dlaset(Upper, m-1, m-1, zero, zero, a.Off(0, 1))
				ie = 1
				itauq = ie + m
				itaup = itauq + m
				nwork = itaup + m

				//              Bidiagonalize L in A
				//              Workspace: need   3*M [e, tauq, taup] + M      [work]
				//              Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [work]
				if err = Dgebrd(m, m, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				nwork = ie + m

				//              Perform bidiagonal SVD, computing singular values only
				//              Workspace: need   M [e] + BDSPAC
				if err = Dbdsdc(Upper, 'N', m, s, work.Off(ie-1), dum.Matrix(1, opts), dum.Matrix(1, opts), dum, &idum, work.Off(nwork-1), iwork); err != nil {
					panic(err)
				}

			} else if wntqo {
				//              Path 2t (N >> M, JOBZ='O')
				//              M right singular vectors to be overwritten on A and
				//              M left singular vectors to be computed in U
				ivt = 1

				//              WORK(IVT) is M by M
				//              WORK(IL)  is M by M; it is later resized to M by chunk for gemm
				il = ivt + m*m
				if lwork >= m*n+m*m+3*m+bdspac {
					ldwrkl = m
					chunk = n
				} else {
					ldwrkl = m
					chunk = (lwork - m*m) / m
				}
				itau = il + ldwrkl*m
				nwork = itau + m

				//              Compute A=L*Q
				//              Workspace: need   M*M [VT] + M*M [L] + M [tau] + M    [work]
				//              Workspace: prefer M*M [VT] + M*M [L] + M [tau] + M*NB [work]
				if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy L to WORK(IL), zeroing about above it
				Dlacpy(Lower, m, m, a, work.MatrixOff(il-1, ldwrkl, opts))
				Dlaset(Upper, m-1, m-1, zero, zero, work.MatrixOff(il+ldwrkl-1, ldwrkl, opts))

				//              Generate Q in A
				//              Workspace: need   M*M [VT] + M*M [L] + M [tau] + M    [work]
				//              Workspace: prefer M*M [VT] + M*M [L] + M [tau] + M*NB [work]
				if err = Dorglq(m, n, m, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				ie = itau
				itauq = ie + m
				itaup = itauq + m
				nwork = itaup + m

				//              Bidiagonalize L in WORK(IL)
				//              Workspace: need   M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + M      [work]
				//              Workspace: prefer M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + 2*M*NB [work]
				if err = Dgebrd(m, m, work.MatrixOff(il-1, ldwrkl, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in U, and computing right singular
				//              vectors of bidiagonal matrix in WORK(IVT)
				//              Workspace: need   M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + BDSPAC
				if err = Dbdsdc(Upper, 'I', m, s, work.Off(ie-1), u, work.MatrixOff(ivt-1, m, opts), dum, &idum, work.Off(nwork-1), iwork); err != nil {
					panic(err)
				}

				//              Overwrite U by left singular vectors of L and WORK(IVT)
				//              by right singular vectors of L
				//              Workspace: need   M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + M    [work]
				//              Workspace: prefer M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + M*NB [work]
				if err = Dormbr('Q', Left, NoTrans, m, m, m, work.MatrixOff(il-1, ldwrkl, opts), work.Off(itauq-1), u, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				if err = Dormbr('P', Right, Trans, m, m, m, work.MatrixOff(il-1, ldwrkl, opts), work.Off(itaup-1), work.MatrixOff(ivt-1, m, opts), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Multiply right singular vectors of L in WORK(IVT) by Q
				//              in A, storing result in WORK(IL) and copying to A
				//              Workspace: need   M*M [VT] + M*M [L]
				//              Workspace: prefer M*M [VT] + M*N [L]
				//              At this point, L is resized as M by chunk.
				for i = 1; i <= n; i += chunk {
					blk = min(n-i+1, chunk)
					err = goblas.Dgemm(NoTrans, NoTrans, m, blk, m, one, work.MatrixOff(ivt-1, m, opts), a.Off(0, i-1), zero, work.MatrixOff(il-1, ldwrkl, opts))
					Dlacpy(Full, m, blk, work.MatrixOff(il-1, ldwrkl, opts), a.Off(0, i-1))
				}

			} else if wntqs {
				//              Path 3t (N >> M, JOBZ='S')
				//              M right singular vectors to be computed in VT and
				//              M left singular vectors to be computed in U
				il = 1

				//              WORK(IL) is M by M
				ldwrkl = m
				itau = il + ldwrkl*m
				nwork = itau + m

				//              Compute A=L*Q
				//              Workspace: need   M*M [L] + M [tau] + M    [work]
				//              Workspace: prefer M*M [L] + M [tau] + M*NB [work]
				if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy L to WORK(IL), zeroing out above it
				Dlacpy(Lower, m, m, a, work.MatrixOff(il-1, ldwrkl, opts))
				Dlaset(Upper, m-1, m-1, zero, zero, work.MatrixOff(il+ldwrkl-1, ldwrkl, opts))

				//              Generate Q in A
				//              Workspace: need   M*M [L] + M [tau] + M    [work]
				//              Workspace: prefer M*M [L] + M [tau] + M*NB [work]
				if err = Dorglq(m, n, m, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				ie = itau
				itauq = ie + m
				itaup = itauq + m
				nwork = itaup + m

				//              Bidiagonalize L in WORK(IU).
				//              Workspace: need   M*M [L] + 3*M [e, tauq, taup] + M      [work]
				//              Workspace: prefer M*M [L] + 3*M [e, tauq, taup] + 2*M*NB [work]
				if err = Dgebrd(m, m, work.MatrixOff(il-1, ldwrkl, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in U and computing right singular
				//              vectors of bidiagonal matrix in VT
				//              Workspace: need   M*M [L] + 3*M [e, tauq, taup] + BDSPAC
				if err = Dbdsdc(Upper, 'I', m, s, work.Off(ie-1), u, vt, dum, &idum, work.Off(nwork-1), iwork); err != nil {
					panic(err)
				}

				//              Overwrite U by left singular vectors of L and VT
				//              by right singular vectors of L
				//              Workspace: need   M*M [L] + 3*M [e, tauq, taup] + M    [work]
				//              Workspace: prefer M*M [L] + 3*M [e, tauq, taup] + M*NB [work]
				if err = Dormbr('Q', Left, NoTrans, m, m, m, work.MatrixOff(il-1, ldwrkl, opts), work.Off(itauq-1), u, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				if err = Dormbr('P', Right, Trans, m, m, m, work.MatrixOff(il-1, ldwrkl, opts), work.Off(itaup-1), vt, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Multiply right singular vectors of L in WORK(IL) by
				//              Q in A, storing result in VT
				//              Workspace: need   M*M [L]
				Dlacpy(Full, m, m, vt, work.MatrixOff(il-1, ldwrkl, opts))
				err = goblas.Dgemm(NoTrans, NoTrans, m, n, m, one, work.MatrixOff(il-1, ldwrkl, opts), a, zero, vt)

			} else if wntqa {
				//              Path 4t (N >> M, JOBZ='A')
				//              N right singular vectors to be computed in VT and
				//              M left singular vectors to be computed in U
				ivt = 1

				//              WORK(IVT) is M by M
				ldwkvt = m
				itau = ivt + ldwkvt*m
				nwork = itau + m

				//              Compute A=L*Q, copying result to VT
				//              Workspace: need   M*M [VT] + M [tau] + M    [work]
				//              Workspace: prefer M*M [VT] + M [tau] + M*NB [work]
				if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				Dlacpy(Upper, m, n, a, vt)

				//              Generate Q in VT
				//              Workspace: need   M*M [VT] + M [tau] + N    [work]
				//              Workspace: prefer M*M [VT] + M [tau] + N*NB [work]
				if err = Dorglq(n, n, m, vt, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Produce L in A, zeroing out other entries
				Dlaset(Upper, m-1, m-1, zero, zero, a.Off(0, 1))
				ie = itau
				itauq = ie + m
				itaup = itauq + m
				nwork = itaup + m

				//              Bidiagonalize L in A
				//              Workspace: need   M*M [VT] + 3*M [e, tauq, taup] + M      [work]
				//              Workspace: prefer M*M [VT] + 3*M [e, tauq, taup] + 2*M*NB [work]
				if err = Dgebrd(m, m, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in U and computing right singular
				//              vectors of bidiagonal matrix in WORK(IVT)
				//              Workspace: need   M*M [VT] + 3*M [e, tauq, taup] + BDSPAC
				if err = Dbdsdc(Upper, 'I', m, s, work.Off(ie-1), u, work.MatrixOff(ivt-1, ldwkvt, opts), dum, &idum, work.Off(nwork-1), iwork); err != nil {
					panic(err)
				}

				//              Overwrite U by left singular vectors of L and WORK(IVT)
				//              by right singular vectors of L
				//              Workspace: need   M*M [VT] + 3*M [e, tauq, taup]+ M    [work]
				//              Workspace: prefer M*M [VT] + 3*M [e, tauq, taup]+ M*NB [work]
				if err = Dormbr('Q', Left, NoTrans, m, m, m, a, work.Off(itauq-1), u, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				if err = Dormbr('P', Right, Trans, m, m, m, a, work.Off(itaup-1), work.MatrixOff(ivt-1, ldwkvt, opts), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Multiply right singular vectors of L in WORK(IVT) by
				//              Q in VT, storing result in A
				//              Workspace: need   M*M [VT]
				err = goblas.Dgemm(NoTrans, NoTrans, m, n, m, one, work.MatrixOff(ivt-1, ldwkvt, opts), vt, zero, a)

				//              Copy right singular vectors of A from A to VT
				Dlacpy(Full, m, n, a, vt)

			}

		} else {
			//           N .LT. MNTHR
			//
			//           Path 5t (N > M, but not much larger)
			//           Reduce to bidiagonal form without LQ decomposition
			ie = 1
			itauq = ie + m
			itaup = itauq + m
			nwork = itaup + m

			//           Bidiagonalize A
			//           Workspace: need   3*M [e, tauq, taup] + N        [work]
			//           Workspace: prefer 3*M [e, tauq, taup] + (M+N)*NB [work]
			if err = Dgebrd(m, n, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
				panic(err)
			}
			if wntqn {
				//              Path 5tn (N > M, JOBZ='N')
				//              Perform bidiagonal SVD, only computing singular values
				//              Workspace: need   3*M [e, tauq, taup] + BDSPAC
				if err = Dbdsdc(Lower, 'N', m, s, work.Off(ie-1), dum.Matrix(1, opts), dum.Matrix(1, opts), dum, &idum, work.Off(nwork-1), iwork); err != nil {
					panic(err)
				}
			} else if wntqo {
				//              Path 5to (N > M, JOBZ='O')
				ldwkvt = m
				ivt = nwork
				if lwork >= m*n+3*m+bdspac {
					//                 WORK( IVT ) is M by N
					Dlaset(Full, m, n, zero, zero, work.MatrixOff(ivt-1, ldwkvt, opts))
					nwork = ivt + ldwkvt*n
					//                 IL is unused; silence compile warnings
					il = -1
				} else {
					//                 WORK( IVT ) is M by M
					nwork = ivt + ldwkvt*m
					il = nwork

					//                 WORK(IL) is M by CHUNK
					chunk = (lwork - m*m - 3*m) / m
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in U and computing right singular
				//              vectors of bidiagonal matrix in WORK(IVT)
				//              Workspace: need   3*M [e, tauq, taup] + M*M [VT] + BDSPAC
				if err = Dbdsdc(Lower, 'I', m, s, work.Off(ie-1), u, work.MatrixOff(ivt-1, ldwkvt, opts), dum, &idum, work.Off(nwork-1), iwork); err != nil {
					panic(err)
				}

				//              Overwrite U by left singular vectors of A
				//              Workspace: need   3*M [e, tauq, taup] + M*M [VT] + M    [work]
				//              Workspace: prefer 3*M [e, tauq, taup] + M*M [VT] + M*NB [work]
				if err = Dormbr('Q', Left, NoTrans, m, m, n, a, work.Off(itauq-1), u, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				if lwork >= m*n+3*m+bdspac {
					//                 Path 5to-fast
					//                 Overwrite WORK(IVT) by left singular vectors of A
					//                 Workspace: need   3*M [e, tauq, taup] + M*N [VT] + M    [work]
					//                 Workspace: prefer 3*M [e, tauq, taup] + M*N [VT] + M*NB [work]
					if err = Dormbr('P', Right, Trans, m, n, m, a, work.Off(itaup-1), work.MatrixOff(ivt-1, ldwkvt, opts), work.Off(nwork-1), lwork-nwork+1); err != nil {
						panic(err)
					}

					//                 Copy right singular vectors of A from WORK(IVT) to A
					Dlacpy(Full, m, n, work.MatrixOff(ivt-1, ldwkvt, opts), a)
				} else {
					//                 Path 5to-slow
					//                 Generate P**T in A
					//                 Workspace: need   3*M [e, tauq, taup] + M*M [VT] + M    [work]
					//                 Workspace: prefer 3*M [e, tauq, taup] + M*M [VT] + M*NB [work]
					if err = Dorgbr('P', m, n, m, a, work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
						panic(err)
					}

					//                 Multiply Q in A by right singular vectors of
					//                 bidiagonal matrix in WORK(IVT), storing result in
					//                 WORK(IL) and copying to A
					//                 Workspace: need   3*M [e, tauq, taup] + M*M [VT] + M*NB [L]
					//                 Workspace: prefer 3*M [e, tauq, taup] + M*M [VT] + M*N  [L]
					for i = 1; i <= n; i += chunk {
						blk = min(n-i+1, chunk)
						err = goblas.Dgemm(NoTrans, NoTrans, m, blk, m, one, work.MatrixOff(ivt-1, ldwkvt, opts), a.Off(0, i-1), zero, work.MatrixOff(il-1, m, opts))
						Dlacpy(Full, m, blk, work.MatrixOff(il-1, m, opts), a.Off(0, i-1))
					}
				}
			} else if wntqs {
				//              Path 5ts (N > M, JOBZ='S')
				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in U and computing right singular
				//              vectors of bidiagonal matrix in VT
				//              Workspace: need   3*M [e, tauq, taup] + BDSPAC
				Dlaset(Full, m, n, zero, zero, vt)
				if err = Dbdsdc(Lower, 'I', m, s, work.Off(ie-1), u, vt, dum, &idum, work.Off(nwork-1), iwork); err != nil {
					panic(err)
				}

				//              Overwrite U by left singular vectors of A and VT
				//              by right singular vectors of A
				//              Workspace: need   3*M [e, tauq, taup] + M    [work]
				//              Workspace: prefer 3*M [e, tauq, taup] + M*NB [work]
				if err = Dormbr('Q', Left, NoTrans, m, m, n, a, work.Off(itauq-1), u, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				if err = Dormbr('P', Right, Trans, m, n, m, a, work.Off(itaup-1), vt, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
			} else if wntqa {
				//              Path 5ta (N > M, JOBZ='A')
				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in U and computing right singular
				//              vectors of bidiagonal matrix in VT
				//              Workspace: need   3*M [e, tauq, taup] + BDSPAC
				Dlaset(Full, n, n, zero, zero, vt)
				if err = Dbdsdc(Lower, 'I', m, s, work.Off(ie-1), u, vt, dum, &idum, work.Off(nwork-1), iwork); err != nil {
					panic(err)
				}

				//              Set the right corner of VT to identity matrix
				if n > m {
					Dlaset(Full, n-m, n-m, zero, one, vt.Off(m, m))
				}

				//              Overwrite U by left singular vectors of A and VT
				//              by right singular vectors of A
				//              Workspace: need   3*M [e, tauq, taup] + N    [work]
				//              Workspace: prefer 3*M [e, tauq, taup] + N*NB [work]
				if err = Dormbr('Q', Left, NoTrans, m, m, n, a, work.Off(itauq-1), u, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				if err = Dormbr('P', Right, Trans, n, n, m, a, work.Off(itaup-1), vt, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
			}

		}

	}

	//     Undo scaling if necessary
	if iscl == 1 {
		if anrm > bignum {
			if err = Dlascl('G', 0, 0, bignum, anrm, minmn, 1, s.Matrix(minmn, opts)); err != nil {
				panic(err)
			}
		}
		if anrm < smlnum {
			if err = Dlascl('G', 0, 0, smlnum, anrm, minmn, 1, s.Matrix(minmn, opts)); err != nil {
				panic(err)
			}
		}
	}

	//     Return optimal workspace in WORK(1)
	work.Set(0, float64(maxwrk))

	return
}
