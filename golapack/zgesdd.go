package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgesdd computes the singular value decomposition (SVD) of a complex
// M-by-N matrix A, optionally computing the left and/or right singular
// vectors, by using divide-and-conquer method. The SVD is written
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
// Note that the routine returns VT = V**H, not V.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Zgesdd(jobz byte, m, n int, a *mat.CMatrix, s *mat.Vector, u, vt *mat.CMatrix, work *mat.CVector, lwork int, rwork *mat.Vector, iwork *[]int) (info int, err error) {
	var lquery, wntqa, wntqas, wntqn, wntqo, wntqs bool
	var cone, czero complex128
	var anrm, bignum, eps, one, smlnum, zero float64
	var blk, chunk, i, ie, il, ir, iru, irvt, iscl, itau, itaup, itauq, iu, ivt, ldwkvt, ldwrkl, ldwrkr, ldwrku, lworkZgebrdMm, lworkZgebrdMn, lworkZgebrdNn, lworkZgelqfMn, lworkZgeqrfMn, lworkZungbrPMn, lworkZungbrPNn, lworkZungbrQMm, lworkZungbrQMn, lworkZunglqMn, lworkZunglqNn, lworkZungqrMm, lworkZungqrMn, lworkZunmbrPrcMm, lworkZunmbrPrcMn, lworkZunmbrPrcNn, lworkZunmbrQlnMm, lworkZunmbrQlnMn, lworkZunmbrQlnNn, maxwrk, minmn, minwrk, mnthr1, mnthr2, nrwork, nwork, wrkbl int

	cdum := cvf(100)
	dum := vf(100)
	idum := make([]int, 1)

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	zero = 0.0
	one = 1.0

	//     Test the input arguments
	minmn = min(m, n)
	mnthr1 = int(float64(minmn) * 17.0 / 9.0)
	mnthr2 = int(float64(minmn) * 5.0 / 3.0)
	wntqa = jobz == 'A'
	wntqs = jobz == 'S'
	wntqas = wntqa || wntqs
	wntqo = jobz == 'O'
	wntqn = jobz == 'N'
	lquery = (lwork == -1)
	minwrk = 1
	maxwrk = 1

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
		err = fmt.Errorf("vt.Rows < 1 || (wntqa && vt.Rows < n) || (wntqs && vt.Rows < minmn) || (wntqo && m >= n && vt.Rows < n): jobz='%c', vt.Rows=%v, m=%v, n=%v", jobz, vt.Rows, m, n)
	}

	//     Compute workspace
	//       Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace allocated at that point in the code,
	//       as well as the preferred amount for good performance.
	//       CWorkspace refers to complex workspace, and RWorkspace to
	//       real workspace. NB refers to the optimal block size for the
	//       immediately following subroutine, as returned by ILAENV.)
	if err == nil {
		minwrk = 1
		maxwrk = 1
		if m >= n && minmn > 0 {
			//           There is no complex work space needed for bidiagonal SVD
			//           The real work space needed for bidiagonal SVD (dbdsdc) is
			//           BDSPAC = 3*N*N + 4*N for singular values and vectors;
			//           BDSPAC = 4*N         for singular values only;
			//           not including e, RU, and RVT matrices.
			//
			//           Compute space preferred for each routine
			if err = Zgebrd(m, n, cdum.CMatrix(m, opts), dum, dum, cdum, cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZgebrdMn = int(cdum.GetRe(0))

			if err = Zgebrd(n, n, cdum.CMatrix(n, opts), dum, dum, cdum, cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZgebrdNn = int(cdum.GetRe(0))

			if err = Zgeqrf(m, n, cdum.CMatrix(m, opts), cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZgeqrfMn = int(cdum.GetRe(0))

			if err = Zungbr('P', n, n, n, cdum.CMatrix(n, opts), cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZungbrPNn = int(cdum.GetRe(0))

			if err = Zungbr('Q', m, m, n, cdum.CMatrix(m, opts), cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZungbrQMm = int(cdum.GetRe(0))

			if err = Zungbr('Q', m, n, n, cdum.CMatrix(m, opts), cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZungbrQMn = int(cdum.GetRe(0))

			if err = Zungqr(m, m, n, cdum.CMatrix(m, opts), cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZungqrMm = int(cdum.GetRe(0))

			if err = Zungqr(m, n, n, cdum.CMatrix(m, opts), cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZungqrMn = int(cdum.GetRe(0))

			if err = Zunmbr('P', Right, ConjTrans, n, n, n, cdum.CMatrix(n, opts), cdum, cdum.CMatrix(n, opts), cdum, -1); err != nil {
				panic(err)
			}
			lworkZunmbrPrcNn = int(cdum.GetRe(0))

			if err = Zunmbr('Q', Left, NoTrans, m, m, n, cdum.CMatrix(m, opts), cdum, cdum.CMatrix(m, opts), cdum, -1); err != nil {
				panic(err)
			}
			lworkZunmbrQlnMm = int(cdum.GetRe(0))

			if err = Zunmbr('Q', Left, NoTrans, m, n, n, cdum.CMatrix(m, opts), cdum, cdum.CMatrix(m, opts), cdum, -1); err != nil {
				panic(err)
			}
			lworkZunmbrQlnMn = int(cdum.GetRe(0))

			if err = Zunmbr('Q', Left, NoTrans, n, n, n, cdum.CMatrix(n, opts), cdum, cdum.CMatrix(m, opts), cdum, -1); err != nil {
				panic(err)
			}
			lworkZunmbrQlnNn = int(cdum.GetRe(0))

			if m >= mnthr1 {
				if wntqn {
					//                 Path 1 (M >> N, JOBZ='N')
					maxwrk = n + lworkZgeqrfMn
					maxwrk = max(maxwrk, 2*n+lworkZgebrdNn)
					minwrk = 3 * n
				} else if wntqo {
					//                 Path 2 (M >> N, JOBZ='O')
					wrkbl = n + lworkZgeqrfMn
					wrkbl = max(wrkbl, n+lworkZungqrMn)
					wrkbl = max(wrkbl, 2*n+lworkZgebrdNn)
					wrkbl = max(wrkbl, 2*n+lworkZunmbrQlnNn)
					wrkbl = max(wrkbl, 2*n+lworkZunmbrPrcNn)
					maxwrk = m*n + n*n + wrkbl
					minwrk = 2*n*n + 3*n
				} else if wntqs {
					//                 Path 3 (M >> N, JOBZ='S')
					wrkbl = n + lworkZgeqrfMn
					wrkbl = max(wrkbl, n+lworkZungqrMn)
					wrkbl = max(wrkbl, 2*n+lworkZgebrdNn)
					wrkbl = max(wrkbl, 2*n+lworkZunmbrQlnNn)
					wrkbl = max(wrkbl, 2*n+lworkZunmbrPrcNn)
					maxwrk = n*n + wrkbl
					minwrk = n*n + 3*n
				} else if wntqa {
					//                 Path 4 (M >> N, JOBZ='A')
					wrkbl = n + lworkZgeqrfMn
					wrkbl = max(wrkbl, n+lworkZungqrMm)
					wrkbl = max(wrkbl, 2*n+lworkZgebrdNn)
					wrkbl = max(wrkbl, 2*n+lworkZunmbrQlnNn)
					wrkbl = max(wrkbl, 2*n+lworkZunmbrPrcNn)
					maxwrk = n*n + wrkbl
					minwrk = n*n + max(3*n, n+m)
				}
			} else if m >= mnthr2 {
				//              Path 5 (M >> N, but not as much as MNTHR1)
				maxwrk = 2*n + lworkZgebrdMn
				minwrk = 2*n + m
				if wntqo {
					//                 Path 5o (M >> N, JOBZ='O')
					maxwrk = max(maxwrk, 2*n+lworkZungbrPNn)
					maxwrk = max(maxwrk, 2*n+lworkZungbrQMn)
					maxwrk = maxwrk + m*n
					minwrk = minwrk + n*n
				} else if wntqs {
					//                 Path 5s (M >> N, JOBZ='S')
					maxwrk = max(maxwrk, 2*n+lworkZungbrPNn)
					maxwrk = max(maxwrk, 2*n+lworkZungbrQMn)
				} else if wntqa {
					//                 Path 5a (M >> N, JOBZ='A')
					maxwrk = max(maxwrk, 2*n+lworkZungbrPNn)
					maxwrk = max(maxwrk, 2*n+lworkZungbrQMm)
				}
			} else {
				//              Path 6 (M >= N, but not much larger)
				maxwrk = 2*n + lworkZgebrdMn
				minwrk = 2*n + m
				if wntqo {
					//                 Path 6o (M >= N, JOBZ='O')
					maxwrk = max(maxwrk, 2*n+lworkZunmbrPrcNn)
					maxwrk = max(maxwrk, 2*n+lworkZunmbrQlnMn)
					maxwrk = maxwrk + m*n
					minwrk = minwrk + n*n
				} else if wntqs {
					//                 Path 6s (M >= N, JOBZ='S')
					maxwrk = max(maxwrk, 2*n+lworkZunmbrQlnMn)
					maxwrk = max(maxwrk, 2*n+lworkZunmbrPrcNn)
				} else if wntqa {
					//                 Path 6a (M >= N, JOBZ='A')
					maxwrk = max(maxwrk, 2*n+lworkZunmbrQlnMm)
					maxwrk = max(maxwrk, 2*n+lworkZunmbrPrcNn)
				}
			}
		} else if minmn > 0 {
			//           There is no complex work space needed for bidiagonal SVD
			//           The real work space needed for bidiagonal SVD (dbdsdc) is
			//           BDSPAC = 3*M*M + 4*M for singular values and vectors;
			//           BDSPAC = 4*M         for singular values only;
			//           not including e, RU, and RVT matrices.
			//
			//           Compute space preferred for each routine
			if err = Zgebrd(m, n, cdum.CMatrix(m, opts), dum, dum, cdum, cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZgebrdMn = int(cdum.GetRe(0))

			if err = Zgebrd(m, m, cdum.CMatrix(m, opts), dum, dum, cdum, cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZgebrdMm = int(cdum.GetRe(0))

			if err = Zgelqf(m, n, cdum.CMatrix(m, opts), cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZgelqfMn = int(cdum.GetRe(0))

			if err = Zungbr('P', m, n, m, cdum.CMatrix(m, opts), cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZungbrPMn = int(cdum.GetRe(0))

			if err = Zungbr('P', n, n, m, cdum.CMatrix(n, opts), cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZungbrPNn = int(cdum.GetRe(0))

			if err = Zungbr('Q', m, m, n, cdum.CMatrix(m, opts), cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZungbrQMm = int(cdum.GetRe(0))

			if err = Zunglq(m, n, m, cdum.CMatrix(m, opts), cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZunglqMn = int(cdum.GetRe(0))

			if err = Zunglq(n, n, m, cdum.CMatrix(n, opts), cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZunglqNn = int(cdum.GetRe(0))

			if err = Zunmbr('P', Right, ConjTrans, m, m, m, cdum.CMatrix(m, opts), cdum, cdum.CMatrix(m, opts), cdum, -1); err != nil {
				panic(err)
			}
			lworkZunmbrPrcMm = int(cdum.GetRe(0))

			if err = Zunmbr('P', Right, ConjTrans, m, n, m, cdum.CMatrix(m, opts), cdum, cdum.CMatrix(m, opts), cdum, -1); err != nil {
				panic(err)
			}
			lworkZunmbrPrcMn = int(cdum.GetRe(0))

			if err = Zunmbr('P', Right, ConjTrans, n, n, m, cdum.CMatrix(n, opts), cdum, cdum.CMatrix(n, opts), cdum, -1); err != nil {
				panic(err)
			}
			lworkZunmbrPrcNn = int(cdum.GetRe(0))

			if err = Zunmbr('Q', Left, NoTrans, m, m, m, cdum.CMatrix(m, opts), cdum, cdum.CMatrix(m, opts), cdum, -1); err != nil {
				panic(err)
			}
			lworkZunmbrQlnMm = int(cdum.GetRe(0))

			if n >= mnthr1 {
				if wntqn {
					//                 Path 1t (N >> M, JOBZ='N')
					maxwrk = m + lworkZgelqfMn
					maxwrk = max(maxwrk, 2*m+lworkZgebrdMm)
					minwrk = 3 * m
				} else if wntqo {
					//                 Path 2t (N >> M, JOBZ='O')
					wrkbl = m + lworkZgelqfMn
					wrkbl = max(wrkbl, m+lworkZunglqMn)
					wrkbl = max(wrkbl, 2*m+lworkZgebrdMm)
					wrkbl = max(wrkbl, 2*m+lworkZunmbrQlnMm)
					wrkbl = max(wrkbl, 2*m+lworkZunmbrPrcMm)
					maxwrk = m*n + m*m + wrkbl
					minwrk = 2*m*m + 3*m
				} else if wntqs {
					//                 Path 3t (N >> M, JOBZ='S')
					wrkbl = m + lworkZgelqfMn
					wrkbl = max(wrkbl, m+lworkZunglqMn)
					wrkbl = max(wrkbl, 2*m+lworkZgebrdMm)
					wrkbl = max(wrkbl, 2*m+lworkZunmbrQlnMm)
					wrkbl = max(wrkbl, 2*m+lworkZunmbrPrcMm)
					maxwrk = m*m + wrkbl
					minwrk = m*m + 3*m
				} else if wntqa {
					//                 Path 4t (N >> M, JOBZ='A')
					wrkbl = m + lworkZgelqfMn
					wrkbl = max(wrkbl, m+lworkZunglqNn)
					wrkbl = max(wrkbl, 2*m+lworkZgebrdMm)
					wrkbl = max(wrkbl, 2*m+lworkZunmbrQlnMm)
					wrkbl = max(wrkbl, 2*m+lworkZunmbrPrcMm)
					maxwrk = m*m + wrkbl
					minwrk = m*m + max(3*m, m+n)
				}
			} else if n >= mnthr2 {
				//              Path 5t (N >> M, but not as much as MNTHR1)
				maxwrk = 2*m + lworkZgebrdMn
				minwrk = 2*m + n
				if wntqo {
					//                 Path 5to (N >> M, JOBZ='O')
					maxwrk = max(maxwrk, 2*m+lworkZungbrQMm)
					maxwrk = max(maxwrk, 2*m+lworkZungbrPMn)
					maxwrk = maxwrk + m*n
					minwrk = minwrk + m*m
				} else if wntqs {
					//                 Path 5ts (N >> M, JOBZ='S')
					maxwrk = max(maxwrk, 2*m+lworkZungbrQMm)
					maxwrk = max(maxwrk, 2*m+lworkZungbrPMn)
				} else if wntqa {
					//                 Path 5ta (N >> M, JOBZ='A')
					maxwrk = max(maxwrk, 2*m+lworkZungbrQMm)
					maxwrk = max(maxwrk, 2*m+lworkZungbrPNn)
				}
			} else {
				//              Path 6t (N > M, but not much larger)
				maxwrk = 2*m + lworkZgebrdMn
				minwrk = 2*m + n
				if wntqo {
					//                 Path 6to (N > M, JOBZ='O')
					maxwrk = max(maxwrk, 2*m+lworkZunmbrQlnMm)
					maxwrk = max(maxwrk, 2*m+lworkZunmbrPrcMn)
					maxwrk = maxwrk + m*n
					minwrk = minwrk + m*m
				} else if wntqs {
					//                 Path 6ts (N > M, JOBZ='S')
					maxwrk = max(maxwrk, 2*m+lworkZunmbrQlnMm)
					maxwrk = max(maxwrk, 2*m+lworkZunmbrPrcMn)
				} else if wntqa {
					//                 Path 6ta (N > M, JOBZ='A')
					maxwrk = max(maxwrk, 2*m+lworkZunmbrQlnMm)
					maxwrk = max(maxwrk, 2*m+lworkZunmbrPrcNn)
				}
			}
		}
		maxwrk = max(maxwrk, minwrk)
	}
	if err == nil {
		work.SetRe(0, float64(maxwrk))
		if lwork < minwrk && !lquery {
			err = fmt.Errorf("lwork < minwrk && !lquery: lwork=%v, minwrk=%v, lquery=%v", lwork, minwrk, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zgesdd", err)
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
	anrm = Zlange('M', m, n, a, dum)
	iscl = 0
	if anrm > zero && anrm < smlnum {
		iscl = 1
		if err = Zlascl('G', 0, 0, anrm, smlnum, m, n, a); err != nil {
			panic(err)
		}
	} else if anrm > bignum {
		iscl = 1
		if err = Zlascl('G', 0, 0, anrm, bignum, m, n, a); err != nil {
			panic(err)
		}
	}

	if m >= n {
		//        A has at least as many rows as columns. If A has sufficiently
		//        more rows than columns, first reduce using the QR
		//        decomposition (if sufficient workspace available)
		if m >= mnthr1 {

			if wntqn {
				//              Path 1 (M >> N, JOBZ='N')
				//              No singular vectors to be computed
				itau = 1
				nwork = itau + n

				//              Compute A=Q*R
				//              CWorkspace: need   N [tau] + N    [work]
				//              CWorkspace: prefer N [tau] + N*NB [work]
				//              RWorkspace: need   0
				if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Zero out below R
				Zlaset(Lower, n-1, n-1, czero, czero, a.Off(1, 0))
				ie = 1
				itauq = 1
				itaup = itauq + n
				nwork = itaup + n

				//              Bidiagonalize R in A
				//              CWorkspace: need   2*N [tauq, taup] + N      [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + 2*N*NB [work]
				//              RWorkspace: need   N [e]
				if err = Zgebrd(n, n, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				nrwork = ie + n

				//              Perform bidiagonal SVD, compute singular values only
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + BDSPAC
				if err = Dbdsdc(Upper, 'N', n, s, rwork.Off(ie-1), dum.Matrix(1, opts), dum.Matrix(1, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

			} else if wntqo {
				//              Path 2 (M >> N, JOBZ='O')
				//              N left singular vectors to be overwritten on A and
				//              N right singular vectors to be computed in VT
				iu = 1

				//              WORK(IU) is N by N
				ldwrku = n
				ir = iu + ldwrku*n
				if lwork >= m*n+n*n+3*n {
					//                 WORK(IR) is M by N
					ldwrkr = m
				} else {
					ldwrkr = (lwork - n*n - 3*n) / n
				}
				itau = ir + ldwrkr*n
				nwork = itau + n

				//              Compute A=Q*R
				//              CWorkspace: need   N*N [U] + N*N [R] + N [tau] + N    [work]
				//              CWorkspace: prefer N*N [U] + N*N [R] + N [tau] + N*NB [work]
				//              RWorkspace: need   0
				if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy R to WORK( IR ), zeroing out below it
				Zlacpy(Upper, n, n, a, work.Off(ir-1).CMatrix(ldwrkr, opts))
				Zlaset(Lower, n-1, n-1, czero, czero, work.Off(ir).CMatrix(ldwrkr, opts))

				//              Generate Q in A
				//              CWorkspace: need   N*N [U] + N*N [R] + N [tau] + N    [work]
				//              CWorkspace: prefer N*N [U] + N*N [R] + N [tau] + N*NB [work]
				//              RWorkspace: need   0
				if err = Zungqr(m, n, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				ie = 1
				itauq = itau
				itaup = itauq + n
				nwork = itaup + n

				//              Bidiagonalize R in WORK(IR)
				//              CWorkspace: need   N*N [U] + N*N [R] + 2*N [tauq, taup] + N      [work]
				//              CWorkspace: prefer N*N [U] + N*N [R] + 2*N [tauq, taup] + 2*N*NB [work]
				//              RWorkspace: need   N [e]
				if err = Zgebrd(n, n, work.Off(ir-1).CMatrix(ldwrkr, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of R in WORK(IRU) and computing right singular vectors
				//              of R in WORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
				iru = ie + n
				irvt = iru + n*n
				nrwork = irvt + n*n
				if err = Dbdsdc(Upper, 'I', n, s, rwork.Off(ie-1), rwork.Off(iru-1).Matrix(n, opts), rwork.Off(irvt-1).Matrix(n, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
				//              Overwrite WORK(IU) by the left singular vectors of R
				//              CWorkspace: need   N*N [U] + N*N [R] + 2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer N*N [U] + N*N [R] + 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacp2(Full, n, n, rwork.Off(iru-1).Matrix(n, opts), work.Off(iu-1).CMatrix(ldwrku, opts))
				if err = Zunmbr('Q', Left, NoTrans, n, n, n, work.Off(ir-1).CMatrix(ldwrkr, opts), work.Off(itauq-1), work.Off(iu-1).CMatrix(ldwrku, opts), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRVT) to complex matrix VT
				//              Overwrite VT by the right singular vectors of R
				//              CWorkspace: need   N*N [U] + N*N [R] + 2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer N*N [U] + N*N [R] + 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacp2(Full, n, n, rwork.Off(irvt-1).Matrix(n, opts), vt)
				if err = Zunmbr('P', Right, ConjTrans, n, n, n, work.Off(ir-1).CMatrix(ldwrkr, opts), work.Off(itaup-1), vt, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Multiply Q in A by left singular vectors of R in
				//              WORK(IU), storing result in WORK(IR) and copying to A
				//              CWorkspace: need   N*N [U] + N*N [R]
				//              CWorkspace: prefer N*N [U] + M*N [R]
				//              RWorkspace: need   0
				for i = 1; i <= m; i += ldwrkr {
					chunk = min(m-i+1, ldwrkr)
					if err = work.Off(ir-1).CMatrix(ldwrkr, opts).Gemm(NoTrans, NoTrans, chunk, n, n, cone, a.Off(i-1, 0), work.Off(iu-1).CMatrix(ldwrku, opts), czero); err != nil {
						panic(err)
					}
					Zlacpy(Full, chunk, n, work.Off(ir-1).CMatrix(ldwrkr, opts), a.Off(i-1, 0))
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
				//              CWorkspace: need   N*N [R] + N [tau] + N    [work]
				//              CWorkspace: prefer N*N [R] + N [tau] + N*NB [work]
				//              RWorkspace: need   0
				if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy R to WORK(IR), zeroing out below it
				Zlacpy(Upper, n, n, a, work.Off(ir-1).CMatrix(ldwrkr, opts))
				Zlaset(Lower, n-1, n-1, czero, czero, work.Off(ir).CMatrix(ldwrkr, opts))

				//              Generate Q in A
				//              CWorkspace: need   N*N [R] + N [tau] + N    [work]
				//              CWorkspace: prefer N*N [R] + N [tau] + N*NB [work]
				//              RWorkspace: need   0
				if err = Zungqr(m, n, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				ie = 1
				itauq = itau
				itaup = itauq + n
				nwork = itaup + n

				//              Bidiagonalize R in WORK(IR)
				//              CWorkspace: need   N*N [R] + 2*N [tauq, taup] + N      [work]
				//              CWorkspace: prefer N*N [R] + 2*N [tauq, taup] + 2*N*NB [work]
				//              RWorkspace: need   N [e]
				if err = Zgebrd(n, n, work.Off(ir-1).CMatrix(ldwrkr, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
				iru = ie + n
				irvt = iru + n*n
				nrwork = irvt + n*n
				if err = Dbdsdc(Upper, 'I', n, s, rwork.Off(ie-1), rwork.Off(iru-1).Matrix(n, opts), rwork.Off(irvt-1).Matrix(n, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRU) to complex matrix U
				//              Overwrite U by left singular vectors of R
				//              CWorkspace: need   N*N [R] + 2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer N*N [R] + 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacp2(Full, n, n, rwork.Off(iru-1).Matrix(n, opts), u)
				if err = Zunmbr('Q', Left, NoTrans, n, n, n, work.Off(ir-1).CMatrix(ldwrkr, opts), work.Off(itauq-1), u, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRVT) to complex matrix VT
				//              Overwrite VT by right singular vectors of R
				//              CWorkspace: need   N*N [R] + 2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer N*N [R] + 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacp2(Full, n, n, rwork.Off(irvt-1).Matrix(n, opts), vt)
				if err = Zunmbr('P', Right, ConjTrans, n, n, n, work.Off(ir-1).CMatrix(ldwrkr, opts), work.Off(itaup-1), vt, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Multiply Q in A by left singular vectors of R in
				//              WORK(IR), storing result in U
				//              CWorkspace: need   N*N [R]
				//              RWorkspace: need   0
				Zlacpy(Full, n, n, u, work.Off(ir-1).CMatrix(ldwrkr, opts))
				if err = u.Gemm(NoTrans, NoTrans, m, n, n, cone, a, work.Off(ir-1).CMatrix(ldwrkr, opts), czero); err != nil {
					panic(err)
				}

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
				//              CWorkspace: need   N*N [U] + N [tau] + N    [work]
				//              CWorkspace: prefer N*N [U] + N [tau] + N*NB [work]
				//              RWorkspace: need   0
				if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				Zlacpy(Lower, m, n, a, u)

				//              Generate Q in U
				//              CWorkspace: need   N*N [U] + N [tau] + M    [work]
				//              CWorkspace: prefer N*N [U] + N [tau] + M*NB [work]
				//              RWorkspace: need   0
				if err = Zungqr(m, m, n, u, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Produce R in A, zeroing out below it
				Zlaset(Lower, n-1, n-1, czero, czero, a.Off(1, 0))
				ie = 1
				itauq = itau
				itaup = itauq + n
				nwork = itaup + n

				//              Bidiagonalize R in A
				//              CWorkspace: need   N*N [U] + 2*N [tauq, taup] + N      [work]
				//              CWorkspace: prefer N*N [U] + 2*N [tauq, taup] + 2*N*NB [work]
				//              RWorkspace: need   N [e]
				if err = Zgebrd(n, n, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				iru = ie + n
				irvt = iru + n*n
				nrwork = irvt + n*n

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
				if err = Dbdsdc(Upper, 'I', n, s, rwork.Off(ie-1), rwork.Off(iru-1).Matrix(n, opts), rwork.Off(irvt-1).Matrix(n, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
				//              Overwrite WORK(IU) by left singular vectors of R
				//              CWorkspace: need   N*N [U] + 2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer N*N [U] + 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacp2(Full, n, n, rwork.Off(iru-1).Matrix(n, opts), work.Off(iu-1).CMatrix(ldwrku, opts))
				if err = Zunmbr('Q', Left, NoTrans, n, n, n, a, work.Off(itauq-1), work.Off(iu-1).CMatrix(ldwrku, opts), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRVT) to complex matrix VT
				//              Overwrite VT by right singular vectors of R
				//              CWorkspace: need   N*N [U] + 2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer N*N [U] + 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacp2(Full, n, n, rwork.Off(irvt-1).Matrix(n, opts), vt)
				if err = Zunmbr('P', Right, ConjTrans, n, n, n, a, work.Off(itaup-1), vt, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Multiply Q in U by left singular vectors of R in
				//              WORK(IU), storing result in A
				//              CWorkspace: need   N*N [U]
				//              RWorkspace: need   0
				if err = a.Gemm(NoTrans, NoTrans, m, n, n, cone, u, work.Off(iu-1).CMatrix(ldwrku, opts), czero); err != nil {
					panic(err)
				}

				//              Copy left singular vectors of A from A to U
				Zlacpy(Full, m, n, a, u)

			}

		} else if m >= mnthr2 {
			//           MNTHR2 <= M < MNTHR1
			//
			//           Path 5 (M >> N, but not as much as MNTHR1)
			//           Reduce to bidiagonal form without QR decomposition, use
			//           ZUNGBR and matrix multiplication to compute singular vectors
			ie = 1
			nrwork = ie + n
			itauq = 1
			itaup = itauq + n
			nwork = itaup + n

			//           Bidiagonalize A
			//           CWorkspace: need   2*N [tauq, taup] + M        [work]
			//           CWorkspace: prefer 2*N [tauq, taup] + (M+N)*NB [work]
			//           RWorkspace: need   N [e]
			if err = Zgebrd(m, n, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
				panic(err)
			}
			if wntqn {
				//              Path 5n (M >> N, JOBZ='N')
				//              Compute singular values only
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + BDSPAC
				if err = Dbdsdc(Upper, 'N', n, s, rwork.Off(ie-1), dum.Matrix(1, opts), dum.Matrix(1, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}
			} else if wntqo {
				iu = nwork
				iru = nrwork
				irvt = iru + n*n
				nrwork = irvt + n*n

				//              Path 5o (M >> N, JOBZ='O')
				//              Copy A to VT, generate P**H
				//              CWorkspace: need   2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacpy(Upper, n, n, a, vt)
				if err = Zungbr('P', n, n, n, vt, work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Generate Q in A
				//              CWorkspace: need   2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				if err = Zungbr('Q', m, n, n, a, work.Off(itauq-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				if lwork >= m*n+3*n {
					//                 WORK( IU ) is M by N
					ldwrku = m
				} else {
					//                 WORK(IU) is LDWRKU by N
					ldwrku = (lwork - 3*n) / n
				}
				nwork = iu + ldwrku*n

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
				if err = Dbdsdc(Upper, 'I', n, s, rwork.Off(ie-1), rwork.Off(iru-1).Matrix(n, opts), rwork.Off(irvt-1).Matrix(n, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

				//              Multiply real matrix RWORK(IRVT) by P**H in VT,
				//              storing the result in WORK(IU), copying to VT
				//              CWorkspace: need   2*N [tauq, taup] + N*N [U]
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + 2*N*N [rwork]
				Zlarcm(n, n, rwork.Off(irvt-1).Matrix(n, opts), vt, work.Off(iu-1).CMatrix(ldwrku, opts), rwork.Off(nrwork-1))
				Zlacpy(Full, n, n, work.Off(iu-1).CMatrix(ldwrku, opts), vt)

				//              Multiply Q in A by real matrix RWORK(IRU), storing the
				//              result in WORK(IU), copying to A
				//              CWorkspace: need   2*N [tauq, taup] + N*N [U]
				//              CWorkspace: prefer 2*N [tauq, taup] + M*N [U]
				//              RWorkspace: need   N [e] + N*N [RU] + 2*N*N [rwork]
				//              RWorkspace: prefer N [e] + N*N [RU] + 2*M*N [rwork] < N + 5*N*N since M < 2*N here
				nrwork = irvt
				for i = 1; i <= m; i += ldwrku {
					chunk = min(m-i+1, ldwrku)
					Zlacrm(chunk, n, a.Off(i-1, 0), rwork.Off(iru-1).Matrix(n, opts), work.Off(iu-1).CMatrix(ldwrku, opts), rwork.Off(nrwork-1))
					Zlacpy(Full, chunk, n, work.Off(iu-1).CMatrix(ldwrku, opts), a.Off(i-1, 0))
				}

			} else if wntqs {
				//              Path 5s (M >> N, JOBZ='S')
				//              Copy A to VT, generate P**H
				//              CWorkspace: need   2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacpy(Upper, n, n, a, vt)
				if err = Zungbr('P', n, n, n, vt, work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy A to U, generate Q
				//              CWorkspace: need   2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacpy(Lower, m, n, a, u)
				if err = Zungbr('Q', m, n, n, u, work.Off(itauq-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
				iru = nrwork
				irvt = iru + n*n
				nrwork = irvt + n*n
				if err = Dbdsdc(Upper, 'I', n, s, rwork.Off(ie-1), rwork.Off(iru-1).Matrix(n, opts), rwork.Off(irvt-1).Matrix(n, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

				//              Multiply real matrix RWORK(IRVT) by P**H in VT,
				//              storing the result in A, copying to VT
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + 2*N*N [rwork]
				Zlarcm(n, n, rwork.Off(irvt-1).Matrix(n, opts), vt, a, rwork.Off(nrwork-1))
				Zlacpy(Full, n, n, a, vt)

				//              Multiply Q in U by real matrix RWORK(IRU), storing the
				//              result in A, copying to U
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + 2*M*N [rwork] < N + 5*N*N since M < 2*N here
				nrwork = irvt
				Zlacrm(m, n, u, rwork.Off(iru-1).Matrix(n, opts), a, rwork.Off(nrwork-1))
				Zlacpy(Full, m, n, a, u)
			} else {
				//              Path 5a (M >> N, JOBZ='A')
				//              Copy A to VT, generate P**H
				//              CWorkspace: need   2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacpy(Upper, n, n, a, vt)
				if err = Zungbr('P', n, n, n, vt, work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy A to U, generate Q
				//              CWorkspace: need   2*N [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacpy(Lower, m, n, a, u)
				if err = Zungbr('Q', m, m, n, u, work.Off(itauq-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
				iru = nrwork
				irvt = iru + n*n
				nrwork = irvt + n*n
				if err = Dbdsdc(Upper, 'I', n, s, rwork.Off(ie-1), rwork.Off(iru-1).Matrix(n, opts), rwork.Off(irvt-1).Matrix(n, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

				//              Multiply real matrix RWORK(IRVT) by P**H in VT,
				//              storing the result in A, copying to VT
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + 2*N*N [rwork]
				Zlarcm(n, n, rwork.Off(irvt-1).Matrix(n, opts), vt, a, rwork.Off(nrwork-1))
				Zlacpy(Full, n, n, a, vt)

				//              Multiply Q in U by real matrix RWORK(IRU), storing the
				//              result in A, copying to U
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + 2*M*N [rwork] < N + 5*N*N since M < 2*N here
				nrwork = irvt
				Zlacrm(m, n, u, rwork.Off(iru-1).Matrix(n, opts), a, rwork.Off(nrwork-1))
				Zlacpy(Full, m, n, a, u)
			}

		} else {
			//           M .LT. MNTHR2
			//
			//           Path 6 (M >= N, but not much larger)
			//           Reduce to bidiagonal form without QR decomposition
			//           Use ZUNMBR to compute singular vectors
			ie = 1
			nrwork = ie + n
			itauq = 1
			itaup = itauq + n
			nwork = itaup + n

			//           Bidiagonalize A
			//           CWorkspace: need   2*N [tauq, taup] + M        [work]
			//           CWorkspace: prefer 2*N [tauq, taup] + (M+N)*NB [work]
			//           RWorkspace: need   N [e]
			if err = Zgebrd(m, n, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
				panic(err)
			}
			if wntqn {
				//              Path 6n (M >= N, JOBZ='N')
				//              Compute singular values only
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + BDSPAC
				if err = Dbdsdc(Upper, 'N', n, s, rwork.Off(ie-1), dum.Matrix(1, opts), dum.Matrix(1, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}
			} else if wntqo {
				iu = nwork
				iru = nrwork
				irvt = iru + n*n
				nrwork = irvt + n*n
				if lwork >= m*n+3*n {
					//                 WORK( IU ) is M by N
					ldwrku = m
				} else {
					//                 WORK( IU ) is LDWRKU by N
					ldwrku = (lwork - 3*n) / n
				}
				nwork = iu + ldwrku*n

				//              Path 6o (M >= N, JOBZ='O')
				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
				if err = Dbdsdc(Upper, 'I', n, s, rwork.Off(ie-1), rwork.Off(iru-1).Matrix(n, opts), rwork.Off(irvt-1).Matrix(n, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRVT) to complex matrix VT
				//              Overwrite VT by right singular vectors of A
				//              CWorkspace: need   2*N [tauq, taup] + N*N [U] + N    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + N*N [U] + N*NB [work]
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
				Zlacp2(Full, n, n, rwork.Off(irvt-1).Matrix(n, opts), vt)
				if err = Zunmbr('P', Right, ConjTrans, n, n, n, a, work.Off(itaup-1), vt, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				if lwork >= m*n+3*n {
					//                 Path 6o-fast
					//                 Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
					//                 Overwrite WORK(IU) by left singular vectors of A, copying
					//                 to A
					//                 CWorkspace: need   2*N [tauq, taup] + M*N [U] + N    [work]
					//                 CWorkspace: prefer 2*N [tauq, taup] + M*N [U] + N*NB [work]
					//                 RWorkspace: need   N [e] + N*N [RU]
					Zlaset(Full, m, n, czero, czero, work.Off(iu-1).CMatrix(ldwrku, opts))
					Zlacp2(Full, n, n, rwork.Off(iru-1).Matrix(n, opts), work.Off(iu-1).CMatrix(ldwrku, opts))
					if err = Zunmbr('Q', Left, NoTrans, m, n, n, a, work.Off(itauq-1), work.Off(iu-1).CMatrix(ldwrku, opts), work.Off(nwork-1), lwork-nwork+1); err != nil {
						panic(err)
					}
					Zlacpy(Full, m, n, work.Off(iu-1).CMatrix(ldwrku, opts), a)
				} else {
					//                 Path 6o-slow
					//                 Generate Q in A
					//                 CWorkspace: need   2*N [tauq, taup] + N*N [U] + N    [work]
					//                 CWorkspace: prefer 2*N [tauq, taup] + N*N [U] + N*NB [work]
					//                 RWorkspace: need   0
					if err = Zungbr('Q', m, n, n, a, work.Off(itauq-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
						panic(err)
					}

					//                 Multiply Q in A by real matrix RWORK(IRU), storing the
					//                 result in WORK(IU), copying to A
					//                 CWorkspace: need   2*N [tauq, taup] + N*N [U]
					//                 CWorkspace: prefer 2*N [tauq, taup] + M*N [U]
					//                 RWorkspace: need   N [e] + N*N [RU] + 2*N*N [rwork]
					//                 RWorkspace: prefer N [e] + N*N [RU] + 2*M*N [rwork] < N + 5*N*N since M < 2*N here
					nrwork = irvt
					for i = 1; i <= m; i += ldwrku {
						chunk = min(m-i+1, ldwrku)
						Zlacrm(chunk, n, a.Off(i-1, 0), rwork.Off(iru-1).Matrix(n, opts), work.Off(iu-1).CMatrix(ldwrku, opts), rwork.Off(nrwork-1))
						Zlacpy(Full, chunk, n, work.Off(iu-1).CMatrix(ldwrku, opts), a.Off(i-1, 0))
					}
				}

			} else if wntqs {
				//              Path 6s (M >= N, JOBZ='S')
				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
				iru = nrwork
				irvt = iru + n*n
				nrwork = irvt + n*n
				if err = Dbdsdc(Upper, 'I', n, s, rwork.Off(ie-1), rwork.Off(iru-1).Matrix(n, opts), rwork.Off(irvt-1).Matrix(n, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRU) to complex matrix U
				//              Overwrite U by left singular vectors of A
				//              CWorkspace: need   2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
				Zlaset(Full, m, n, czero, czero, u)
				Zlacp2(Full, n, n, rwork.Off(iru-1).Matrix(n, opts), u)
				if err = Zunmbr('Q', Left, NoTrans, m, n, n, a, work.Off(itauq-1), u, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRVT) to complex matrix VT
				//              Overwrite VT by right singular vectors of A
				//              CWorkspace: need   2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
				Zlacp2(Full, n, n, rwork.Off(irvt-1).Matrix(n, opts), vt)
				if err = Zunmbr('P', Right, ConjTrans, n, n, n, a, work.Off(itaup-1), vt, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
			} else {
				//              Path 6a (M >= N, JOBZ='A')
				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
				iru = nrwork
				irvt = iru + n*n
				nrwork = irvt + n*n
				if err = Dbdsdc(Upper, 'I', n, s, rwork.Off(ie-1), rwork.Off(iru-1).Matrix(n, opts), rwork.Off(irvt-1).Matrix(n, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

				//              Set the right corner of U to identity matrix
				Zlaset(Full, m, m, czero, czero, u)
				if m > n {
					Zlaset(Full, m-n, m-n, czero, cone, u.Off(n, n))
				}

				//              Copy real matrix RWORK(IRU) to complex matrix U
				//              Overwrite U by left singular vectors of A
				//              CWorkspace: need   2*N [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + M*NB [work]
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
				Zlacp2(Full, n, n, rwork.Off(iru-1).Matrix(n, opts), u)
				if err = Zunmbr('Q', Left, NoTrans, m, m, n, a, work.Off(itauq-1), u, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRVT) to complex matrix VT
				//              Overwrite VT by right singular vectors of A
				//              CWorkspace: need   2*N [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
				//              RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
				Zlacp2(Full, n, n, rwork.Off(irvt-1).Matrix(n, opts), vt)
				if err = Zunmbr('P', Right, ConjTrans, n, n, n, a, work.Off(itaup-1), vt, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
			}

		}

	} else {
		//        A has more columns than rows. If A has sufficiently more
		//        columns than rows, first reduce using the LQ decomposition (if
		//        sufficient workspace available)
		if n >= mnthr1 {

			if wntqn {
				//              Path 1t (N >> M, JOBZ='N')
				//              No singular vectors to be computed
				itau = 1
				nwork = itau + m

				//              Compute A=L*Q
				//              CWorkspace: need   M [tau] + M    [work]
				//              CWorkspace: prefer M [tau] + M*NB [work]
				//              RWorkspace: need   0
				if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Zero out above L
				Zlaset(Upper, m-1, m-1, czero, czero, a.Off(0, 1))
				ie = 1
				itauq = 1
				itaup = itauq + m
				nwork = itaup + m

				//              Bidiagonalize L in A
				//              CWorkspace: need   2*M [tauq, taup] + M      [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + 2*M*NB [work]
				//              RWorkspace: need   M [e]
				if err = Zgebrd(m, m, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				nrwork = ie + m

				//              Perform bidiagonal SVD, compute singular values only
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + BDSPAC
				if err = Dbdsdc(Upper, 'N', m, s, rwork.Off(ie-1), dum.Matrix(1, opts), dum.Matrix(1, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

			} else if wntqo {
				//              Path 2t (N >> M, JOBZ='O')
				//              M right singular vectors to be overwritten on A and
				//              M left singular vectors to be computed in U
				ivt = 1
				ldwkvt = m

				//              WORK(IVT) is M by M
				il = ivt + ldwkvt*m
				if lwork >= m*n+m*m+3*m {
					//                 WORK(IL) M by N
					ldwrkl = m
					chunk = n
				} else {
					//                 WORK(IL) is M by CHUNK
					ldwrkl = m
					chunk = (lwork - m*m - 3*m) / m
				}
				itau = il + ldwrkl*chunk
				nwork = itau + m

				//              Compute A=L*Q
				//              CWorkspace: need   M*M [VT] + M*M [L] + M [tau] + M    [work]
				//              CWorkspace: prefer M*M [VT] + M*M [L] + M [tau] + M*NB [work]
				//              RWorkspace: need   0
				if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy L to WORK(IL), zeroing about above it
				Zlacpy(Lower, m, m, a, work.Off(il-1).CMatrix(ldwrkl, opts))
				Zlaset(Upper, m-1, m-1, czero, czero, work.Off(il+ldwrkl-1).CMatrix(ldwrkl, opts))

				//              Generate Q in A
				//              CWorkspace: need   M*M [VT] + M*M [L] + M [tau] + M    [work]
				//              CWorkspace: prefer M*M [VT] + M*M [L] + M [tau] + M*NB [work]
				//              RWorkspace: need   0
				if err = Zunglq(m, n, m, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				ie = 1
				itauq = itau
				itaup = itauq + m
				nwork = itaup + m

				//              Bidiagonalize L in WORK(IL)
				//              CWorkspace: need   M*M [VT] + M*M [L] + 2*M [tauq, taup] + M      [work]
				//              CWorkspace: prefer M*M [VT] + M*M [L] + 2*M [tauq, taup] + 2*M*NB [work]
				//              RWorkspace: need   M [e]
				if err = Zgebrd(m, m, work.Off(il-1).CMatrix(ldwrkl, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RU] + M*M [RVT] + BDSPAC
				iru = ie + m
				irvt = iru + m*m
				nrwork = irvt + m*m
				if err = Dbdsdc(Upper, 'I', m, s, rwork.Off(ie-1), rwork.Off(iru-1).Matrix(m, opts), rwork.Off(irvt-1).Matrix(m, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
				//              Overwrite WORK(IU) by the left singular vectors of L
				//              CWorkspace: need   M*M [VT] + M*M [L] + 2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer M*M [VT] + M*M [L] + 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacp2(Full, m, m, rwork.Off(iru-1).Matrix(m, opts), u)
				if err = Zunmbr('Q', Left, NoTrans, m, m, m, work.Off(il-1).CMatrix(ldwrkl, opts), work.Off(itauq-1), u, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT)
				//              Overwrite WORK(IVT) by the right singular vectors of L
				//              CWorkspace: need   M*M [VT] + M*M [L] + 2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer M*M [VT] + M*M [L] + 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacp2(Full, m, m, rwork.Off(irvt-1).Matrix(m, opts), work.Off(ivt-1).CMatrix(ldwkvt, opts))
				if err = Zunmbr('P', Right, ConjTrans, m, m, m, work.Off(il-1).CMatrix(ldwrkl, opts), work.Off(itaup-1), work.Off(ivt-1).CMatrix(ldwkvt, opts), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Multiply right singular vectors of L in WORK(IL) by Q
				//              in A, storing result in WORK(IL) and copying to A
				//              CWorkspace: need   M*M [VT] + M*M [L]
				//              CWorkspace: prefer M*M [VT] + M*N [L]
				//              RWorkspace: need   0
				for i = 1; i <= n; i += chunk {
					blk = min(n-i+1, chunk)
					err = work.Off(il-1).CMatrix(ldwrkl, opts).Gemm(NoTrans, NoTrans, m, blk, m, cone, work.Off(ivt-1).CMatrix(m, opts), a.Off(0, i-1), czero)
					Zlacpy(Full, m, blk, work.Off(il-1).CMatrix(ldwrkl, opts), a.Off(0, i-1))
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
				//              CWorkspace: need   M*M [L] + M [tau] + M    [work]
				//              CWorkspace: prefer M*M [L] + M [tau] + M*NB [work]
				//              RWorkspace: need   0
				if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy L to WORK(IL), zeroing out above it
				Zlacpy(Lower, m, m, a, work.Off(il-1).CMatrix(ldwrkl, opts))
				Zlaset(Upper, m-1, m-1, czero, czero, work.Off(il+ldwrkl-1).CMatrix(ldwrkl, opts))

				//              Generate Q in A
				//              CWorkspace: need   M*M [L] + M [tau] + M    [work]
				//              CWorkspace: prefer M*M [L] + M [tau] + M*NB [work]
				//              RWorkspace: need   0
				if err = Zunglq(m, n, m, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				ie = 1
				itauq = itau
				itaup = itauq + m
				nwork = itaup + m

				//              Bidiagonalize L in WORK(IL)
				//              CWorkspace: need   M*M [L] + 2*M [tauq, taup] + M      [work]
				//              CWorkspace: prefer M*M [L] + 2*M [tauq, taup] + 2*M*NB [work]
				//              RWorkspace: need   M [e]
				if err = Zgebrd(m, m, work.Off(il-1).CMatrix(ldwrkl, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RU] + M*M [RVT] + BDSPAC
				iru = ie + m
				irvt = iru + m*m
				nrwork = irvt + m*m
				if err = Dbdsdc(Upper, 'I', m, s, rwork.Off(ie-1), rwork.Off(iru-1).Matrix(m, opts), rwork.Off(irvt-1).Matrix(m, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRU) to complex matrix U
				//              Overwrite U by left singular vectors of L
				//              CWorkspace: need   M*M [L] + 2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer M*M [L] + 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacp2(Full, m, m, rwork.Off(iru-1).Matrix(m, opts), u)
				if err = Zunmbr('Q', Left, NoTrans, m, m, m, work.Off(il-1).CMatrix(ldwrkl, opts), work.Off(itauq-1), u, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRVT) to complex matrix VT
				//              Overwrite VT by left singular vectors of L
				//              CWorkspace: need   M*M [L] + 2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer M*M [L] + 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacp2(Full, m, m, rwork.Off(irvt-1).Matrix(m, opts), vt)
				if err = Zunmbr('P', Right, ConjTrans, m, m, m, work.Off(il-1).CMatrix(ldwrkl, opts), work.Off(itaup-1), vt, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy VT to WORK(IL), multiply right singular vectors of L
				//              in WORK(IL) by Q in A, storing result in VT
				//              CWorkspace: need   M*M [L]
				//              RWorkspace: need   0
				Zlacpy(Full, m, m, vt, work.Off(il-1).CMatrix(ldwrkl, opts))
				if err = vt.Gemm(NoTrans, NoTrans, m, n, m, cone, work.Off(il-1).CMatrix(ldwrkl, opts), a, czero); err != nil {
					panic(err)
				}

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
				//              CWorkspace: need   M*M [VT] + M [tau] + M    [work]
				//              CWorkspace: prefer M*M [VT] + M [tau] + M*NB [work]
				//              RWorkspace: need   0
				if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
				Zlacpy(Upper, m, n, a, vt)

				//              Generate Q in VT
				//              CWorkspace: need   M*M [VT] + M [tau] + N    [work]
				//              CWorkspace: prefer M*M [VT] + M [tau] + N*NB [work]
				//              RWorkspace: need   0
				if err = Zunglq(n, n, m, vt, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Produce L in A, zeroing out above it
				Zlaset(Upper, m-1, m-1, czero, czero, a.Off(0, 1))
				ie = 1
				itauq = itau
				itaup = itauq + m
				nwork = itaup + m

				//              Bidiagonalize L in A
				//              CWorkspace: need   M*M [VT] + 2*M [tauq, taup] + M      [work]
				//              CWorkspace: prefer M*M [VT] + 2*M [tauq, taup] + 2*M*NB [work]
				//              RWorkspace: need   M [e]
				if err = Zgebrd(m, m, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RU] + M*M [RVT] + BDSPAC
				iru = ie + m
				irvt = iru + m*m
				nrwork = irvt + m*m
				if err = Dbdsdc(Upper, 'I', m, s, rwork.Off(ie-1), rwork.Off(iru-1).Matrix(m, opts), rwork.Off(irvt-1).Matrix(m, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRU) to complex matrix U
				//              Overwrite U by left singular vectors of L
				//              CWorkspace: need   M*M [VT] + 2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer M*M [VT] + 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacp2(Full, m, m, rwork.Off(iru-1).Matrix(m, opts), u)
				if err = Zunmbr('Q', Left, NoTrans, m, m, m, a, work.Off(itauq-1), u, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT)
				//              Overwrite WORK(IVT) by right singular vectors of L
				//              CWorkspace: need   M*M [VT] + 2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer M*M [VT] + 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacp2(Full, m, m, rwork.Off(irvt-1).Matrix(m, opts), work.Off(ivt-1).CMatrix(ldwkvt, opts))
				if err = Zunmbr('P', Right, ConjTrans, m, m, m, a, work.Off(itaup-1), work.Off(ivt-1).CMatrix(ldwkvt, opts), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Multiply right singular vectors of L in WORK(IVT) by
				//              Q in VT, storing result in A
				//              CWorkspace: need   M*M [VT]
				//              RWorkspace: need   0
				if err = a.Gemm(NoTrans, NoTrans, m, n, m, cone, work.Off(ivt-1).CMatrix(ldwkvt, opts), vt, czero); err != nil {
					panic(err)
				}

				//              Copy right singular vectors of A from A to VT
				Zlacpy(Full, m, n, a, vt)

			}

		} else if n >= mnthr2 {
			//           MNTHR2 <= N < MNTHR1
			//
			//           Path 5t (N >> M, but not as much as MNTHR1)
			//           Reduce to bidiagonal form without QR decomposition, use
			//           ZUNGBR and matrix multiplication to compute singular vectors
			ie = 1
			nrwork = ie + m
			itauq = 1
			itaup = itauq + m
			nwork = itaup + m

			//           Bidiagonalize A
			//           CWorkspace: need   2*M [tauq, taup] + N        [work]
			//           CWorkspace: prefer 2*M [tauq, taup] + (M+N)*NB [work]
			//           RWorkspace: need   M [e]
			if err = Zgebrd(m, n, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
				panic(err)
			}

			if wntqn {
				//              Path 5tn (N >> M, JOBZ='N')
				//              Compute singular values only
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + BDSPAC
				if err = Dbdsdc(Lower, 'N', m, s, rwork.Off(ie-1), dum.Matrix(1, opts), dum.Matrix(1, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}
			} else if wntqo {
				irvt = nrwork
				iru = irvt + m*m
				nrwork = iru + m*m
				ivt = nwork

				//              Path 5to (N >> M, JOBZ='O')
				//              Copy A to U, generate Q
				//              CWorkspace: need   2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacpy(Lower, m, m, a, u)
				if err = Zungbr('Q', m, m, n, u, work.Off(itauq-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Generate P**H in A
				//              CWorkspace: need   2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				if err = Zungbr('P', m, n, m, a, work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				ldwkvt = m
				if lwork >= m*n+3*m {
					//                 WORK( IVT ) is M by N
					nwork = ivt + ldwkvt*n
					chunk = n
				} else {
					//                 WORK( IVT ) is M by CHUNK
					chunk = (lwork - 3*m) / m
					nwork = ivt + ldwkvt*chunk
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + BDSPAC
				if err = Dbdsdc(Lower, 'I', m, s, rwork.Off(ie-1), rwork.Off(iru-1).Matrix(m, opts), rwork.Off(irvt-1).Matrix(m, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

				//              Multiply Q in U by real matrix RWORK(IRVT)
				//              storing the result in WORK(IVT), copying to U
				//              CWorkspace: need   2*M [tauq, taup] + M*M [VT]
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + 2*M*M [rwork]
				Zlacrm(m, m, u, rwork.Off(iru-1).Matrix(m, opts), work.Off(ivt-1).CMatrix(ldwkvt, opts), rwork.Off(nrwork-1))
				Zlacpy(Full, m, m, work.Off(ivt-1).CMatrix(ldwkvt, opts), u)

				//              Multiply RWORK(IRVT) by P**H in A, storing the
				//              result in WORK(IVT), copying to A
				//              CWorkspace: need   2*M [tauq, taup] + M*M [VT]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*N [VT]
				//              RWorkspace: need   M [e] + M*M [RVT] + 2*M*M [rwork]
				//              RWorkspace: prefer M [e] + M*M [RVT] + 2*M*N [rwork] < M + 5*M*M since N < 2*M here
				nrwork = iru
				for i = 1; i <= n; i += chunk {
					blk = min(n-i+1, chunk)
					Zlarcm(m, blk, rwork.Off(irvt-1).Matrix(m, opts), a.Off(0, i-1), work.Off(ivt-1).CMatrix(ldwkvt, opts), rwork.Off(nrwork-1))
					Zlacpy(Full, m, blk, work.Off(ivt-1).CMatrix(ldwkvt, opts), a.Off(0, i-1))
				}
			} else if wntqs {
				//              Path 5ts (N >> M, JOBZ='S')
				//              Copy A to U, generate Q
				//              CWorkspace: need   2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacpy(Lower, m, m, a, u)
				if err = Zungbr('Q', m, m, n, u, work.Off(itauq-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy A to VT, generate P**H
				//              CWorkspace: need   2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacpy(Upper, m, n, a, vt)
				if err = Zungbr('P', m, n, m, vt, work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + BDSPAC
				irvt = nrwork
				iru = irvt + m*m
				nrwork = iru + m*m
				if err = Dbdsdc(Lower, 'I', m, s, rwork.Off(ie-1), rwork.Off(iru-1).Matrix(m, opts), rwork.Off(irvt-1).Matrix(m, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

				//              Multiply Q in U by real matrix RWORK(IRU), storing the
				//              result in A, copying to U
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + 2*M*M [rwork]
				Zlacrm(m, m, u, rwork.Off(iru-1).Matrix(m, opts), a, rwork.Off(nrwork-1))
				Zlacpy(Full, m, m, a, u)

				//              Multiply real matrix RWORK(IRVT) by P**H in VT,
				//              storing the result in A, copying to VT
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + 2*M*N [rwork] < M + 5*M*M since N < 2*M here
				nrwork = iru
				Zlarcm(m, n, rwork.Off(irvt-1).Matrix(m, opts), vt, a, rwork.Off(nrwork-1))
				Zlacpy(Full, m, n, a, vt)
			} else {
				//              Path 5ta (N >> M, JOBZ='A')
				//              Copy A to U, generate Q
				//              CWorkspace: need   2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   0
				Zlacpy(Lower, m, m, a, u)
				if err = Zungbr('Q', m, m, n, u, work.Off(itauq-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy A to VT, generate P**H
				//              CWorkspace: need   2*M [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + N*NB [work]
				//              RWorkspace: need   0
				Zlacpy(Upper, m, n, a, vt)
				if err = Zungbr('P', n, n, m, vt, work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + BDSPAC
				irvt = nrwork
				iru = irvt + m*m
				nrwork = iru + m*m
				if err = Dbdsdc(Lower, 'I', m, s, rwork.Off(ie-1), rwork.Off(iru-1).Matrix(m, opts), rwork.Off(irvt-1).Matrix(m, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

				//              Multiply Q in U by real matrix RWORK(IRU), storing the
				//              result in A, copying to U
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + 2*M*M [rwork]
				Zlacrm(m, m, u, rwork.Off(iru-1).Matrix(m, opts), a, rwork.Off(nrwork-1))
				Zlacpy(Full, m, m, a, u)

				//              Multiply real matrix RWORK(IRVT) by P**H in VT,
				//              storing the result in A, copying to VT
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + 2*M*N [rwork] < M + 5*M*M since N < 2*M here
				nrwork = iru
				Zlarcm(m, n, rwork.Off(irvt-1).Matrix(m, opts), vt, a, rwork.Off(nrwork-1))
				Zlacpy(Full, m, n, a, vt)
			}

		} else {
			//           N .LT. MNTHR2
			//
			//           Path 6t (N > M, but not much larger)
			//           Reduce to bidiagonal form without LQ decomposition
			//           Use ZUNMBR to compute singular vectors
			ie = 1
			nrwork = ie + m
			itauq = 1
			itaup = itauq + m
			nwork = itaup + m

			//           Bidiagonalize A
			//           CWorkspace: need   2*M [tauq, taup] + N        [work]
			//           CWorkspace: prefer 2*M [tauq, taup] + (M+N)*NB [work]
			//           RWorkspace: need   M [e]
			if err = Zgebrd(m, n, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
				panic(err)
			}
			if wntqn {
				//              Path 6tn (N > M, JOBZ='N')
				//              Compute singular values only
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + BDSPAC
				if err = Dbdsdc(Lower, 'N', m, s, rwork.Off(ie-1), dum.Matrix(1, opts), dum.Matrix(1, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}
			} else if wntqo {
				//              Path 6to (N > M, JOBZ='O')
				ldwkvt = m
				ivt = nwork
				if lwork >= m*n+3*m {
					//                 WORK( IVT ) is M by N
					Zlaset(Full, m, n, czero, czero, work.Off(ivt-1).CMatrix(ldwkvt, opts))
					nwork = ivt + ldwkvt*n
				} else {
					//                 WORK( IVT ) is M by CHUNK
					chunk = (lwork - 3*m) / m
					nwork = ivt + ldwkvt*chunk
				}

				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + BDSPAC
				irvt = nrwork
				iru = irvt + m*m
				nrwork = iru + m*m
				if err = Dbdsdc(Lower, 'I', m, s, rwork.Off(ie-1), rwork.Off(iru-1).Matrix(m, opts), rwork.Off(irvt-1).Matrix(m, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRU) to complex matrix U
				//              Overwrite U by left singular vectors of A
				//              CWorkspace: need   2*M [tauq, taup] + M*M [VT] + M    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*M [VT] + M*NB [work]
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU]
				Zlacp2(Full, m, m, rwork.Off(iru-1).Matrix(m, opts), u)
				if err = Zunmbr('Q', Left, NoTrans, m, m, n, a, work.Off(itauq-1), u, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				if lwork >= m*n+3*m {
					//                 Path 6to-fast
					//                 Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT)
					//                 Overwrite WORK(IVT) by right singular vectors of A,
					//                 copying to A
					//                 CWorkspace: need   2*M [tauq, taup] + M*N [VT] + M    [work]
					//                 CWorkspace: prefer 2*M [tauq, taup] + M*N [VT] + M*NB [work]
					//                 RWorkspace: need   M [e] + M*M [RVT]
					Zlacp2(Full, m, m, rwork.Off(irvt-1).Matrix(m, opts), work.Off(ivt-1).CMatrix(ldwkvt, opts))
					if err = Zunmbr('P', Right, ConjTrans, m, n, m, a, work.Off(itaup-1), work.Off(ivt-1).CMatrix(ldwkvt, opts), work.Off(nwork-1), lwork-nwork+1); err != nil {
						panic(err)
					}
					Zlacpy(Full, m, n, work.Off(ivt-1).CMatrix(ldwkvt, opts), a)
				} else {
					//                 Path 6to-slow
					//                 Generate P**H in A
					//                 CWorkspace: need   2*M [tauq, taup] + M*M [VT] + M    [work]
					//                 CWorkspace: prefer 2*M [tauq, taup] + M*M [VT] + M*NB [work]
					//                 RWorkspace: need   0
					if err = Zungbr('P', m, n, m, a, work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
						panic(err)
					}

					//                 Multiply Q in A by real matrix RWORK(IRU), storing the
					//                 result in WORK(IU), copying to A
					//                 CWorkspace: need   2*M [tauq, taup] + M*M [VT]
					//                 CWorkspace: prefer 2*M [tauq, taup] + M*N [VT]
					//                 RWorkspace: need   M [e] + M*M [RVT] + 2*M*M [rwork]
					//                 RWorkspace: prefer M [e] + M*M [RVT] + 2*M*N [rwork] < M + 5*M*M since N < 2*M here
					nrwork = iru
					for i = 1; i <= n; i += chunk {
						blk = min(n-i+1, chunk)
						Zlarcm(m, blk, rwork.Off(irvt-1).Matrix(m, opts), a.Off(0, i-1), work.Off(ivt-1).CMatrix(ldwkvt, opts), rwork.Off(nrwork-1))
						Zlacpy(Full, m, blk, work.Off(ivt-1).CMatrix(ldwkvt, opts), a.Off(0, i-1))
					}
				}
			} else if wntqs {
				//              Path 6ts (N > M, JOBZ='S')
				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + BDSPAC
				irvt = nrwork
				iru = irvt + m*m
				nrwork = iru + m*m
				if err = Dbdsdc(Lower, 'I', m, s, rwork.Off(ie-1), rwork.Off(iru-1).Matrix(m, opts), rwork.Off(irvt-1).Matrix(m, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRU) to complex matrix U
				//              Overwrite U by left singular vectors of A
				//              CWorkspace: need   2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU]
				Zlacp2(Full, m, m, rwork.Off(iru-1).Matrix(m, opts), u)
				if err = Zunmbr('Q', Left, NoTrans, m, m, n, a, work.Off(itauq-1), u, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRVT) to complex matrix VT
				//              Overwrite VT by right singular vectors of A
				//              CWorkspace: need   2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   M [e] + M*M [RVT]
				Zlaset(Full, m, n, czero, czero, vt)
				Zlacp2(Full, m, m, rwork.Off(irvt-1).Matrix(m, opts), vt)
				if err = Zunmbr('P', Right, ConjTrans, m, n, m, a, work.Off(itaup-1), vt, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}
			} else {
				//              Path 6ta (N > M, JOBZ='A')
				//              Perform bidiagonal SVD, computing left singular vectors
				//              of bidiagonal matrix in RWORK(IRU) and computing right
				//              singular vectors of bidiagonal matrix in RWORK(IRVT)
				//              CWorkspace: need   0
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + BDSPAC
				irvt = nrwork
				iru = irvt + m*m
				nrwork = iru + m*m

				if err = Dbdsdc(Lower, 'I', m, s, rwork.Off(ie-1), rwork.Off(iru-1).Matrix(m, opts), rwork.Off(irvt-1).Matrix(m, opts), dum, &idum, rwork.Off(nrwork-1), iwork); err != nil {
					panic(err)
				}

				//              Copy real matrix RWORK(IRU) to complex matrix U
				//              Overwrite U by left singular vectors of A
				//              CWorkspace: need   2*M [tauq, taup] + M    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
				//              RWorkspace: need   M [e] + M*M [RVT] + M*M [RU]
				Zlacp2(Full, m, m, rwork.Off(iru-1).Matrix(m, opts), u)
				if err = Zunmbr('Q', Left, NoTrans, m, m, n, a, work.Off(itauq-1), u, work.Off(nwork-1), lwork-nwork+1); err != nil {
					panic(err)
				}

				//              Set all of VT to identity matrix
				Zlaset(Full, n, n, czero, cone, vt)

				//              Copy real matrix RWORK(IRVT) to complex matrix VT
				//              Overwrite VT by right singular vectors of A
				//              CWorkspace: need   2*M [tauq, taup] + N    [work]
				//              CWorkspace: prefer 2*M [tauq, taup] + N*NB [work]
				//              RWorkspace: need   M [e] + M*M [RVT]
				Zlacp2(Full, m, m, rwork.Off(irvt-1).Matrix(m, opts), vt)
				if err = Zunmbr('P', Right, ConjTrans, n, n, m, a, work.Off(itaup-1), vt, work.Off(nwork-1), lwork-nwork+1); err != nil {
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
		if info != 0 && anrm > bignum {
			if err = Dlascl('G', 0, 0, bignum, anrm, minmn-1, 1, rwork.Off(ie-1).Matrix(minmn, opts)); err != nil {
				panic(err)
			}
		}
		if anrm < smlnum {
			if err = Dlascl('G', 0, 0, smlnum, anrm, minmn, 1, s.Matrix(minmn, opts)); err != nil {
				panic(err)
			}
		}
		if info != 0 && anrm < smlnum {
			if err = Dlascl('G', 0, 0, smlnum, anrm, minmn-1, 1, rwork.Off(ie-1).Matrix(minmn, opts)); err != nil {
				panic(err)
			}
		}
	}

	//     Return optimal workspace in WORK(1)
	work.SetRe(0, float64(maxwrk))

	return
}
