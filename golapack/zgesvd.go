package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgesvd computes the singular value decomposition (SVD) of a complex
// M-by-N matrix A, optionally computing the left and/or right singular
// vectors. The SVD is written
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
// Note that the routine returns V**H, not V.
func Zgesvd(jobu, jobvt byte, m, n int, a *mat.CMatrix, s *mat.Vector, u, vt *mat.CMatrix, work *mat.CVector, lwork int, rwork *mat.Vector) (info int, err error) {
	var lquery, wntua, wntuas, wntun, wntuo, wntus, wntva, wntvas, wntvn, wntvo, wntvs bool
	var cone, czero complex128
	var anrm, bignum, eps, one, smlnum, zero float64
	var blk, chunk, i, ie, ir, irwork, iscl, itau, itaup, itauq, iu, iwork, ldwrkr, ldwrku, lworkZgebrd, lworkZgelqf, lworkZgeqrf, lworkZungbrP, lworkZungbrQ, lworkZunglqM, lworkZunglqN, lworkZungqrM, lworkZungqrN, maxwrk, minmn, minwrk, mnthr, ncu, ncvt, nru, nrvt, wrkbl int

	cdum := cvf(1)
	dum := vf(1)

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	zero = 0.0
	one = 1.0

	//     Test the input arguments
	minmn = min(m, n)
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
	lquery = (lwork == -1)

	if !(wntua || wntus || wntuo || wntun) {
		err = fmt.Errorf("!(wntua || wntus || wntuo || wntun): jobu='%c'", jobu)
	} else if !(wntva || wntvs || wntvo || wntvn) || (wntvo && wntuo) {
		err = fmt.Errorf("!(wntva || wntvs || wntvo || wntvn) || (wntvo && wntuo): jobvt='%c'", jobvt)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if u.Rows < 1 || (wntuas && u.Rows < m) {
		err = fmt.Errorf("u.Rows < 1 || (wntuas && u.Rows < m): jobu='%c', u.Rows=%v, m=%v", jobu, u.Rows, m)
	} else if vt.Rows < 1 || (wntva && vt.Rows < n) || (wntvs && vt.Rows < minmn) {
		err = fmt.Errorf("vt.Rows < 1 || (wntva && vt.Rows < n) || (wntvs && vt.Rows < minmn): jobvt='%c', vt.Rows=%v, n=%v", jobvt, vt.Rows, n)
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       CWorkspace refers to complex workspace, and RWorkspace to
	//       real workspace. NB refers to the optimal block size for the
	//       immediately following subroutine, as returned by ILAENV.)
	if err == nil {
		minwrk = 1
		maxwrk = 1
		if m >= n && minmn > 0 {
			//           Space needed for ZBDSQR is BDSPAC = 5*N
			mnthr = Ilaenv(6, "Zgesvd", []byte{jobu, jobvt}, m, n, 0, 0)
			//           Compute space needed for ZGEQRF
			if err = Zgeqrf(m, n, a, cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZgeqrf = int(cdum.GetRe(0))
			//           Compute space needed for ZUNGQR
			if err = Zungqr(m, n, n, a, cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZungqrN = int(cdum.GetRe(0))
			if err = Zungqr(m, m, n, a, cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZungqrM = int(cdum.GetRe(0))
			//           Compute space needed for ZGEBRD
			if err = Zgebrd(n, n, a, s, dum, cdum, cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZgebrd = int(cdum.GetRe(0))
			//           Compute space needed for ZUNGBR
			if err = Zungbr('P', n, n, n, a, cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZungbrP = int(cdum.GetRe(0))
			if err = Zungbr('Q', n, n, n, a, cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZungbrQ = int(cdum.GetRe(0))

			if m >= mnthr {
				if wntun {
					//                 Path 1 (M much larger than N, JOBU='N')
					maxwrk = n + lworkZgeqrf
					maxwrk = max(maxwrk, 2*n+lworkZgebrd)
					if wntvo || wntvas {
						maxwrk = max(maxwrk, 2*n+lworkZungbrP)
					}
					minwrk = 3 * n
				} else if wntuo && wntvn {
					//                 Path 2 (M much larger than N, JOBU='O', JOBVT='N')
					wrkbl = n + lworkZgeqrf
					wrkbl = max(wrkbl, n+lworkZungqrN)
					wrkbl = max(wrkbl, 2*n+lworkZgebrd)
					wrkbl = max(wrkbl, 2*n+lworkZungbrQ)
					maxwrk = max(n*n+wrkbl, n*n+m*n)
					minwrk = 2*n + m
				} else if wntuo && wntvas {
					//                 Path 3 (M much larger than N, JOBU='O', JOBVT='S' or
					//                 'A')
					wrkbl = n + lworkZgeqrf
					wrkbl = max(wrkbl, n+lworkZungqrN)
					wrkbl = max(wrkbl, 2*n+lworkZgebrd)
					wrkbl = max(wrkbl, 2*n+lworkZungbrQ)
					wrkbl = max(wrkbl, 2*n+lworkZungbrP)
					maxwrk = max(n*n+wrkbl, n*n+m*n)
					minwrk = 2*n + m
				} else if wntus && wntvn {
					//                 Path 4 (M much larger than N, JOBU='S', JOBVT='N')
					wrkbl = n + lworkZgeqrf
					wrkbl = max(wrkbl, n+lworkZungqrN)
					wrkbl = max(wrkbl, 2*n+lworkZgebrd)
					wrkbl = max(wrkbl, 2*n+lworkZungbrQ)
					maxwrk = n*n + wrkbl
					minwrk = 2*n + m
				} else if wntus && wntvo {
					//                 Path 5 (M much larger than N, JOBU='S', JOBVT='O')
					wrkbl = n + lworkZgeqrf
					wrkbl = max(wrkbl, n+lworkZungqrN)
					wrkbl = max(wrkbl, 2*n+lworkZgebrd)
					wrkbl = max(wrkbl, 2*n+lworkZungbrQ)
					wrkbl = max(wrkbl, 2*n+lworkZungbrP)
					maxwrk = 2*n*n + wrkbl
					minwrk = 2*n + m
				} else if wntus && wntvas {
					//                 Path 6 (M much larger than N, JOBU='S', JOBVT='S' or
					//                 'A')
					wrkbl = n + lworkZgeqrf
					wrkbl = max(wrkbl, n+lworkZungqrN)
					wrkbl = max(wrkbl, 2*n+lworkZgebrd)
					wrkbl = max(wrkbl, 2*n+lworkZungbrQ)
					wrkbl = max(wrkbl, 2*n+lworkZungbrP)
					maxwrk = n*n + wrkbl
					minwrk = 2*n + m
				} else if wntua && wntvn {
					//                 Path 7 (M much larger than N, JOBU='A', JOBVT='N')
					wrkbl = n + lworkZgeqrf
					wrkbl = max(wrkbl, n+lworkZungqrM)
					wrkbl = max(wrkbl, 2*n+lworkZgebrd)
					wrkbl = max(wrkbl, 2*n+lworkZungbrQ)
					maxwrk = n*n + wrkbl
					minwrk = 2*n + m
				} else if wntua && wntvo {
					//                 Path 8 (M much larger than N, JOBU='A', JOBVT='O')
					wrkbl = n + lworkZgeqrf
					wrkbl = max(wrkbl, n+lworkZungqrM)
					wrkbl = max(wrkbl, 2*n+lworkZgebrd)
					wrkbl = max(wrkbl, 2*n+lworkZungbrQ)
					wrkbl = max(wrkbl, 2*n+lworkZungbrP)
					maxwrk = 2*n*n + wrkbl
					minwrk = 2*n + m
				} else if wntua && wntvas {
					//                 Path 9 (M much larger than N, JOBU='A', JOBVT='S' or
					//                 'A')
					wrkbl = n + lworkZgeqrf
					wrkbl = max(wrkbl, n+lworkZungqrM)
					wrkbl = max(wrkbl, 2*n+lworkZgebrd)
					wrkbl = max(wrkbl, 2*n+lworkZungbrQ)
					wrkbl = max(wrkbl, 2*n+lworkZungbrP)
					maxwrk = n*n + wrkbl
					minwrk = 2*n + m
				}
			} else {
				//              Path 10 (M at least N, but not much larger)
				if err = Zgebrd(m, n, a, s, dum, cdum, cdum, cdum, -1); err != nil {
					panic(err)
				}
				lworkZgebrd = int(cdum.GetRe(0))
				maxwrk = 2*n + lworkZgebrd
				if wntus || wntuo {
					if err = Zungbr('Q', m, n, n, a, cdum, cdum, -1); err != nil {
						panic(err)
					}
					lworkZungbrQ = int(cdum.GetRe(0))
					maxwrk = max(maxwrk, 2*n+lworkZungbrQ)
				}
				if wntua {
					if err = Zungbr('Q', m, m, n, a, cdum, cdum, -1); err != nil {
						panic(err)
					}
					lworkZungbrQ = int(cdum.GetRe(0))
					maxwrk = max(maxwrk, 2*n+lworkZungbrQ)
				}
				if !wntvn {
					maxwrk = max(maxwrk, 2*n+lworkZungbrP)
				}
				minwrk = 2*n + m
			}
		} else if minmn > 0 {
			//           Space needed for ZBDSQR is BDSPAC = 5*M
			mnthr = Ilaenv(6, "Zgesvd", []byte{jobu, jobvt}, m, n, 0, 0)
			//           Compute space needed for ZGELQF
			if err = Zgelqf(m, n, a, cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZgelqf = int(cdum.GetRe(0))
			//           Compute space needed for ZUNGLQ
			if err = Zunglq(n, n, m, cdum.CMatrix(n, opts), cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZunglqN = int(cdum.GetRe(0))
			if err = Zunglq(m, n, m, a, cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZunglqM = int(cdum.GetRe(0))
			//           Compute space needed for ZGEBRD
			if err = Zgebrd(m, m, a, s, dum, cdum, cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZgebrd = int(cdum.GetRe(0))
			//            Compute space needed for ZUNGBR P
			if err = Zungbr('P', m, m, m, a, cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZungbrP = int(cdum.GetRe(0))
			//           Compute space needed for ZUNGBR Q
			if err = Zungbr('Q', m, m, m, a, cdum, cdum, -1); err != nil {
				panic(err)
			}
			lworkZungbrQ = int(cdum.GetRe(0))
			if n >= mnthr {
				if wntvn {
					//                 Path 1t(N much larger than M, JOBVT='N')
					maxwrk = m + lworkZgelqf
					maxwrk = max(maxwrk, 2*m+lworkZgebrd)
					if wntuo || wntuas {
						maxwrk = max(maxwrk, 2*m+lworkZungbrQ)
					}
					minwrk = 3 * m
				} else if wntvo && wntun {
					//                 Path 2t(N much larger than M, JOBU='N', JOBVT='O')
					wrkbl = m + lworkZgelqf
					wrkbl = max(wrkbl, m+lworkZunglqM)
					wrkbl = max(wrkbl, 2*m+lworkZgebrd)
					wrkbl = max(wrkbl, 2*m+lworkZungbrP)
					maxwrk = max(m*m+wrkbl, m*m+m*n)
					minwrk = 2*m + n
				} else if wntvo && wntuas {
					//                 Path 3t(N much larger than M, JOBU='S' or 'A',
					//                 JOBVT='O')
					wrkbl = m + lworkZgelqf
					wrkbl = max(wrkbl, m+lworkZunglqM)
					wrkbl = max(wrkbl, 2*m+lworkZgebrd)
					wrkbl = max(wrkbl, 2*m+lworkZungbrP)
					wrkbl = max(wrkbl, 2*m+lworkZungbrQ)
					maxwrk = max(m*m+wrkbl, m*m+m*n)
					minwrk = 2*m + n
				} else if wntvs && wntun {
					//                 Path 4t(N much larger than M, JOBU='N', JOBVT='S')
					wrkbl = m + lworkZgelqf
					wrkbl = max(wrkbl, m+lworkZunglqM)
					wrkbl = max(wrkbl, 2*m+lworkZgebrd)
					wrkbl = max(wrkbl, 2*m+lworkZungbrP)
					maxwrk = m*m + wrkbl
					minwrk = 2*m + n
				} else if wntvs && wntuo {
					//                 Path 5t(N much larger than M, JOBU='O', JOBVT='S')
					wrkbl = m + lworkZgelqf
					wrkbl = max(wrkbl, m+lworkZunglqM)
					wrkbl = max(wrkbl, 2*m+lworkZgebrd)
					wrkbl = max(wrkbl, 2*m+lworkZungbrP)
					wrkbl = max(wrkbl, 2*m+lworkZungbrQ)
					maxwrk = 2*m*m + wrkbl
					minwrk = 2*m + n
				} else if wntvs && wntuas {
					//                 Path 6t(N much larger than M, JOBU='S' or 'A',
					//                 JOBVT='S')
					wrkbl = m + lworkZgelqf
					wrkbl = max(wrkbl, m+lworkZunglqM)
					wrkbl = max(wrkbl, 2*m+lworkZgebrd)
					wrkbl = max(wrkbl, 2*m+lworkZungbrP)
					wrkbl = max(wrkbl, 2*m+lworkZungbrQ)
					maxwrk = m*m + wrkbl
					minwrk = 2*m + n
				} else if wntva && wntun {
					//                 Path 7t(N much larger than M, JOBU='N', JOBVT='A')
					wrkbl = m + lworkZgelqf
					wrkbl = max(wrkbl, m+lworkZunglqN)
					wrkbl = max(wrkbl, 2*m+lworkZgebrd)
					wrkbl = max(wrkbl, 2*m+lworkZungbrP)
					maxwrk = m*m + wrkbl
					minwrk = 2*m + n
				} else if wntva && wntuo {
					//                 Path 8t(N much larger than M, JOBU='O', JOBVT='A')
					wrkbl = m + lworkZgelqf
					wrkbl = max(wrkbl, m+lworkZunglqN)
					wrkbl = max(wrkbl, 2*m+lworkZgebrd)
					wrkbl = max(wrkbl, 2*m+lworkZungbrP)
					wrkbl = max(wrkbl, 2*m+lworkZungbrQ)
					maxwrk = 2*m*m + wrkbl
					minwrk = 2*m + n
				} else if wntva && wntuas {
					//                 Path 9t(N much larger than M, JOBU='S' or 'A',
					//                 JOBVT='A')
					wrkbl = m + lworkZgelqf
					wrkbl = max(wrkbl, m+lworkZunglqN)
					wrkbl = max(wrkbl, 2*m+lworkZgebrd)
					wrkbl = max(wrkbl, 2*m+lworkZungbrP)
					wrkbl = max(wrkbl, 2*m+lworkZungbrQ)
					maxwrk = m*m + wrkbl
					minwrk = 2*m + n
				}
			} else {
				//              Path 10t(N greater than M, but not much larger)
				if err = Zgebrd(m, n, a, s, dum, cdum, cdum, cdum, -1); err != nil {
					panic(err)
				}
				lworkZgebrd = int(cdum.GetRe(0))
				maxwrk = 2*m + lworkZgebrd
				if wntvs || wntvo {
					//                Compute space needed for ZUNGBR P
					if err = Zungbr('P', m, n, m, a, cdum, cdum, -1); err != nil {
						panic(err)
					}
					lworkZungbrP = int(cdum.GetRe(0))
					maxwrk = max(maxwrk, 2*m+lworkZungbrP)
				}
				if wntva {
					if err = Zungbr('P', n, n, m, a, cdum, cdum, -1); err != nil {
						panic(err)
					}
					lworkZungbrP = int(cdum.GetRe(0))
					maxwrk = max(maxwrk, 2*m+lworkZungbrP)
				}
				if !wntun {
					maxwrk = max(maxwrk, 2*m+lworkZungbrQ)
				}
				minwrk = 2*m + n
			}
		}
		maxwrk = max(maxwrk, minwrk)
		work.SetRe(0, float64(maxwrk))

		if lwork < minwrk && !lquery {
			err = fmt.Errorf("lwork < minwrk && !lquery: lwork=%v, minwrk=%v, lquery=%v", lwork, minwrk, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zgesvd", err)
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
		if m >= mnthr {

			if wntun {
				//              Path 1 (M much larger than N, JOBU='N')
				//              No left singular vectors to be computed
				itau = 1
				iwork = itau + n

				//              Compute A=Q*R
				//              (CWorkspace: need 2*N, prefer N+N*NB)
				//              (RWorkspace: need 0)
				if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}

				//              Zero out below R
				if n > 1 {
					Zlaset(Lower, n-1, n-1, czero, czero, a.Off(1, 0))
				}
				ie = 1
				itauq = 1
				itaup = itauq + n
				iwork = itaup + n

				//              Bidiagonalize R in A
				//              (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
				//              (RWorkspace: need N)
				if err = Zgebrd(n, n, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
				ncvt = 0
				if wntvo || wntvas {
					//                 If right singular vectors desired, generate P'.
					//                 (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
					//                 (RWorkspace: 0)
					if err = Zungbr('P', n, n, n, a, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					ncvt = n
				}
				irwork = ie + n

				//              Perform bidiagonal QR iteration, computing right
				//              singular vectors of A in A if desired
				//              (CWorkspace: 0)
				//              (RWorkspace: need BDSPAC)
				if info, err = Zbdsqr(Upper, n, ncvt, 0, 0, s, rwork.Off(ie-1), a, cdum.CMatrix(1, opts), cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
					panic(err)
				}

				//              If right singular vectors desired in VT, copy them there
				if wntvas {
					Zlacpy(Full, n, n, a, vt)
				}

			} else if wntuo && wntvn {
				//              Path 2 (M much larger than N, JOBU='O', JOBVT='N')
				//              N left singular vectors to be overwritten on A and
				//              no right singular vectors to be computed
				if lwork >= n*n+3*n {
					//                 Sufficient workspace for a fast algorithm
					ir = 1
					if lwork >= max(wrkbl, a.Rows*n)+a.Rows*n {
						//                    WORK(IU) is LDA by N, WORK(IR) is LDA by N
						ldwrku = a.Rows
						ldwrkr = a.Rows
					} else if lwork >= max(wrkbl, a.Rows*n)+n*n {
						//                    WORK(IU) is LDA by N, WORK(IR) is N by N
						ldwrku = a.Rows
						ldwrkr = n
					} else {
						//                    WORK(IU) is LDWRKU by N, WORK(IR) is N by N
						ldwrku = (lwork - n*n) / n
						ldwrkr = n
					}
					itau = ir + ldwrkr*n
					iwork = itau + n

					//                 Compute A=Q*R
					//                 (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
					//                 (RWorkspace: 0)
					if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Copy R to WORK(IR) and zero out below it
					Zlacpy(Upper, n, n, a, work.Off(ir-1).CMatrix(ldwrkr, opts))
					Zlaset(Lower, n-1, n-1, czero, czero, work.Off(ir).CMatrix(ldwrkr, opts))

					//                 Generate Q in A
					//                 (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
					//                 (RWorkspace: 0)
					if err = Zungqr(m, n, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					ie = 1
					itauq = itau
					itaup = itauq + n
					iwork = itaup + n

					//                 Bidiagonalize R in WORK(IR)
					//                 (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
					//                 (RWorkspace: need N)
					if err = Zgebrd(n, n, work.Off(ir-1).CMatrix(ldwrkr, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Generate left vectors bidiagonalizing R
					//                 (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
					//                 (RWorkspace: need 0)
					if err = Zungbr('Q', n, n, n, work.Off(ir-1).CMatrix(ldwrkr, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					irwork = ie + n

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of R in WORK(IR)
					//                 (CWorkspace: need N*N)
					//                 (RWorkspace: need BDSPAC)
					if info, err = Zbdsqr(Upper, n, 0, n, 0, s, rwork.Off(ie-1), cdum.CMatrix(1, opts), work.Off(ir-1).CMatrix(ldwrkr, opts), cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
						panic(err)
					}
					iu = itauq

					//                 Multiply Q in A by left singular vectors of R in
					//                 WORK(IR), storing result in WORK(IU) and copying to A
					//                 (CWorkspace: need N*N+N, prefer N*N+M*N)
					//                 (RWorkspace: 0)
					for i = 1; i <= m; i += ldwrku {
						chunk = min(m-i+1, ldwrku)
						if err = work.Off(iu-1).CMatrix(ldwrku, opts).Gemm(NoTrans, NoTrans, chunk, n, n, cone, a.Off(i-1, 0), work.Off(ir-1).CMatrix(ldwrkr, opts), czero); err != nil {
							panic(err)
						}
						Zlacpy(Full, chunk, n, work.Off(iu-1).CMatrix(ldwrku, opts), a.Off(i-1, 0))
					}

				} else {
					//                 Insufficient workspace for a fast algorithm
					ie = 1
					itauq = 1
					itaup = itauq + n
					iwork = itaup + n

					//                 Bidiagonalize A
					//                 (CWorkspace: need 2*N+M, prefer 2*N+(M+N)*NB)
					//                 (RWorkspace: N)
					if err = Zgebrd(m, n, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Generate left vectors bidiagonalizing A
					//                 (CWorkspace: need 3*N, prefer 2*N+N*NB)
					//                 (RWorkspace: 0)
					if err = Zungbr('Q', m, n, n, a, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					irwork = ie + n

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of A in A
					//                 (CWorkspace: need 0)
					//                 (RWorkspace: need BDSPAC)
					if info, err = Zbdsqr(Upper, n, 0, m, 0, s, rwork.Off(ie-1), cdum.CMatrix(1, opts), a, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
						panic(err)
					}

				}

			} else if wntuo && wntvas {
				//              Path 3 (M much larger than N, JOBU='O', JOBVT='S' or 'A')
				//              N left singular vectors to be overwritten on A and
				//              N right singular vectors to be computed in VT
				if lwork >= n*n+3*n {
					//                 Sufficient workspace for a fast algorithm
					ir = 1
					if lwork >= max(wrkbl, a.Rows*n)+a.Rows*n {
						//                    WORK(IU) is LDA by N and WORK(IR) is LDA by N
						ldwrku = a.Rows
						ldwrkr = a.Rows
					} else if lwork >= max(wrkbl, a.Rows*n)+n*n {
						//                    WORK(IU) is LDA by N and WORK(IR) is N by N
						ldwrku = a.Rows
						ldwrkr = n
					} else {
						//                    WORK(IU) is LDWRKU by N and WORK(IR) is N by N
						ldwrku = (lwork - n*n) / n
						ldwrkr = n
					}
					itau = ir + ldwrkr*n
					iwork = itau + n

					//                 Compute A=Q*R
					//                 (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
					//                 (RWorkspace: 0)
					if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Copy R to VT, zeroing out below it
					Zlacpy(Upper, n, n, a, vt)
					if n > 1 {
						Zlaset(Lower, n-1, n-1, czero, czero, vt.Off(1, 0))
					}

					//                 Generate Q in A
					//                 (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
					//                 (RWorkspace: 0)
					if err = Zungqr(m, n, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					ie = 1
					itauq = itau
					itaup = itauq + n
					iwork = itaup + n

					//                 Bidiagonalize R in VT, copying result to WORK(IR)
					//                 (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
					//                 (RWorkspace: need N)
					if err = Zgebrd(n, n, vt, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					Zlacpy(Lower, n, n, vt, work.Off(ir-1).CMatrix(ldwrkr, opts))

					//                 Generate left vectors bidiagonalizing R in WORK(IR)
					//                 (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
					//                 (RWorkspace: 0)
					if err = Zungbr('Q', n, n, n, work.Off(ir-1).CMatrix(ldwrkr, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Generate right vectors bidiagonalizing R in VT
					//                 (CWorkspace: need N*N+3*N-1, prefer N*N+2*N+(N-1)*NB)
					//                 (RWorkspace: 0)
					if err = Zungbr('P', n, n, n, vt, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					irwork = ie + n

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of R in WORK(IR) and computing right
					//                 singular vectors of R in VT
					//                 (CWorkspace: need N*N)
					//                 (RWorkspace: need BDSPAC)
					if info, err = Zbdsqr(Upper, n, n, n, 0, s, rwork.Off(ie-1), vt, work.Off(ir-1).CMatrix(ldwrkr, opts), cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
						panic(err)
					}
					iu = itauq

					//                 Multiply Q in A by left singular vectors of R in
					//                 WORK(IR), storing result in WORK(IU) and copying to A
					//                 (CWorkspace: need N*N+N, prefer N*N+M*N)
					//                 (RWorkspace: 0)
					for i = 1; i <= m; i += ldwrku {
						chunk = min(m-i+1, ldwrku)
						if err = work.Off(iu-1).CMatrix(ldwrku, opts).Gemm(NoTrans, NoTrans, chunk, n, n, cone, a.Off(i-1, 0), work.Off(ir-1).CMatrix(ldwrkr, opts), czero); err != nil {
							panic(err)
						}
						Zlacpy(Full, chunk, n, work.Off(iu-1).CMatrix(ldwrku, opts), a.Off(i-1, 0))
					}

				} else {
					//                 Insufficient workspace for a fast algorithm
					itau = 1
					iwork = itau + n

					//                 Compute A=Q*R
					//                 (CWorkspace: need 2*N, prefer N+N*NB)
					//                 (RWorkspace: 0)
					if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Copy R to VT, zeroing out below it
					Zlacpy(Upper, n, n, a, vt)
					if n > 1 {
						Zlaset(Lower, n-1, n-1, czero, czero, vt.Off(1, 0))
					}

					//                 Generate Q in A
					//                 (CWorkspace: need 2*N, prefer N+N*NB)
					//                 (RWorkspace: 0)
					if err = Zungqr(m, n, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					ie = 1
					itauq = itau
					itaup = itauq + n
					iwork = itaup + n

					//                 Bidiagonalize R in VT
					//                 (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
					//                 (RWorkspace: N)
					if err = Zgebrd(n, n, vt, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Multiply Q in A by left vectors bidiagonalizing R
					//                 (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
					//                 (RWorkspace: 0)
					if err = Zunmbr('Q', Right, NoTrans, m, n, n, vt, work.Off(itauq-1), a, work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Generate right vectors bidiagonalizing R in VT
					//                 (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
					//                 (RWorkspace: 0)
					if err = Zungbr('P', n, n, n, vt, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					irwork = ie + n

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of A in A and computing right
					//                 singular vectors of A in VT
					//                 (CWorkspace: 0)
					//                 (RWorkspace: need BDSPAC)
					if info, err = Zbdsqr(Upper, n, n, m, 0, s, rwork.Off(ie-1), vt, a, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
						panic(err)
					}

				}

			} else if wntus {

				if wntvn {
					//                 Path 4 (M much larger than N, JOBU='S', JOBVT='N')
					//                 N left singular vectors to be computed in U and
					//                 no right singular vectors to be computed
					if lwork >= n*n+3*n {
						//                    Sufficient workspace for a fast algorithm
						ir = 1
						if lwork >= wrkbl+a.Rows*n {
							//                       WORK(IR) is LDA by N
							ldwrkr = a.Rows
						} else {
							//                       WORK(IR) is N by N
							ldwrkr = n
						}
						itau = ir + ldwrkr*n
						iwork = itau + n

						//                    Compute A=Q*R
						//                    (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy R to WORK(IR), zeroing out below it
						Zlacpy(Upper, n, n, a, work.Off(ir-1).CMatrix(ldwrkr, opts))
						Zlaset(Lower, n-1, n-1, czero, czero, work.Off(ir).CMatrix(ldwrkr, opts))

						//                    Generate Q in A
						//                    (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zungqr(m, n, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = 1
						itauq = itau
						itaup = itauq + n
						iwork = itaup + n

						//                    Bidiagonalize R in WORK(IR)
						//                    (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
						//                    (RWorkspace: need N)
						if err = Zgebrd(n, n, work.Off(ir-1).CMatrix(ldwrkr, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left vectors bidiagonalizing R in WORK(IR)
						//                    (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('Q', n, n, n, work.Off(ir-1).CMatrix(ldwrkr, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IR)
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, n, 0, n, 0, s, rwork.Off(ie-1), cdum.CMatrix(1, opts), work.Off(ir-1).CMatrix(ldwrkr, opts), cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

						//                    Multiply Q in A by left singular vectors of R in
						//                    WORK(IR), storing result in U
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: 0)
						if err = u.Gemm(NoTrans, NoTrans, m, n, n, cone, a, work.Off(ir-1).CMatrix(ldwrkr, opts), czero); err != nil {
							panic(err)
						}

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + n

						//                    Compute A=Q*R, copying result to U
						//                    (CWorkspace: need 2*N, prefer N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Lower, m, n, a, u)

						//                    Generate Q in U
						//                    (CWorkspace: need 2*N, prefer N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zungqr(m, n, n, u, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = 1
						itauq = itau
						itaup = itauq + n
						iwork = itaup + n

						//                    Zero out below R in A
						if n > 1 {
							Zlaset(Lower, n-1, n-1, czero, czero, a.Off(1, 0))
						}

						//                    Bidiagonalize R in A
						//                    (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
						//                    (RWorkspace: need N)
						if err = Zgebrd(n, n, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply Q in U by left vectors bidiagonalizing R
						//                    (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
						//                    (RWorkspace: 0)
						if err = Zunmbr('Q', Right, NoTrans, m, n, n, a, work.Off(itauq-1), u, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, n, 0, m, 0, s, rwork.Off(ie-1), cdum.CMatrix(1, opts), u, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

					}

				} else if wntvo {
					//                 Path 5 (M much larger than N, JOBU='S', JOBVT='O')
					//                 N left singular vectors to be computed in U and
					//                 N right singular vectors to be overwritten on A
					if lwork >= 2*n*n+3*n {
						//                    Sufficient workspace for a fast algorithm
						iu = 1
						if lwork >= wrkbl+2*a.Rows*n {
							//                       WORK(IU) is LDA by N and WORK(IR) is LDA by N
							ldwrku = a.Rows
							ir = iu + ldwrku*n
							ldwrkr = a.Rows
						} else if lwork >= wrkbl+(a.Rows+n)*n {
							//                       WORK(IU) is LDA by N and WORK(IR) is N by N
							ldwrku = a.Rows
							ir = iu + ldwrku*n
							ldwrkr = n
						} else {
							//                       WORK(IU) is N by N and WORK(IR) is N by N
							ldwrku = n
							ir = iu + ldwrku*n
							ldwrkr = n
						}
						itau = ir + ldwrkr*n
						iwork = itau + n

						//                    Compute A=Q*R
						//                    (CWorkspace: need 2*N*N+2*N, prefer 2*N*N+N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy R to WORK(IU), zeroing out below it
						Zlacpy(Upper, n, n, a, work.Off(iu-1).CMatrix(ldwrku, opts))
						Zlaset(Lower, n-1, n-1, czero, czero, work.Off(iu).CMatrix(ldwrku, opts))

						//                    Generate Q in A
						//                    (CWorkspace: need 2*N*N+2*N, prefer 2*N*N+N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zungqr(m, n, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = 1
						itauq = itau
						itaup = itauq + n
						iwork = itaup + n

						//                    Bidiagonalize R in WORK(IU), copying result to
						//                    WORK(IR)
						//                    (CWorkspace: need   2*N*N+3*N,
						//                                 prefer 2*N*N+2*N+2*N*NB)
						//                    (RWorkspace: need   N)
						if err = Zgebrd(n, n, work.Off(iu-1).CMatrix(ldwrku, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Upper, n, n, work.Off(iu-1).CMatrix(ldwrku, opts), work.Off(ir-1).CMatrix(ldwrkr, opts))

						//                    Generate left bidiagonalizing vectors in WORK(IU)
						//                    (CWorkspace: need 2*N*N+3*N, prefer 2*N*N+2*N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('Q', n, n, n, work.Off(iu-1).CMatrix(ldwrku, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right bidiagonalizing vectors in WORK(IR)
						//                    (CWorkspace: need   2*N*N+3*N-1,
						//                                 prefer 2*N*N+2*N+(N-1)*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('P', n, n, n, work.Off(ir-1).CMatrix(ldwrkr, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IU) and computing
						//                    right singular vectors of R in WORK(IR)
						//                    (CWorkspace: need 2*N*N)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, n, n, n, 0, s, rwork.Off(ie-1), work.Off(ir-1).CMatrix(ldwrkr, opts), work.Off(iu-1).CMatrix(ldwrku, opts), cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

						//                    Multiply Q in A by left singular vectors of R in
						//                    WORK(IU), storing result in U
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: 0)
						if err = u.Gemm(NoTrans, NoTrans, m, n, n, cone, a, work.Off(iu-1).CMatrix(ldwrku, opts), czero); err != nil {
							panic(err)
						}

						//                    Copy right singular vectors of R to A
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: 0)
						Zlacpy(Full, n, n, work.Off(ir-1).CMatrix(ldwrkr, opts), a)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + n

						//                    Compute A=Q*R, copying result to U
						//                    (CWorkspace: need 2*N, prefer N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Lower, m, n, a, u)

						//                    Generate Q in U
						//                    (CWorkspace: need 2*N, prefer N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zungqr(m, n, n, u, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = 1
						itauq = itau
						itaup = itauq + n
						iwork = itaup + n

						//                    Zero out below R in A
						if n > 1 {
							Zlaset(Lower, n-1, n-1, czero, czero, a.Off(1, 0))
						}

						//                    Bidiagonalize R in A
						//                    (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
						//                    (RWorkspace: need N)
						if err = Zgebrd(n, n, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply Q in U by left vectors bidiagonalizing R
						//                    (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
						//                    (RWorkspace: 0)
						if err = Zunmbr('Q', Right, NoTrans, m, n, n, a, work.Off(itauq-1), u, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right vectors bidiagonalizing R in A
						//                    (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('P', n, n, n, a, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in A
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, n, n, m, 0, s, rwork.Off(ie-1), a, u, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

					}

				} else if wntvas {
					//                 Path 6 (M much larger than N, JOBU='S', JOBVT='S'
					//                         or 'A')
					//                 N left singular vectors to be computed in U and
					//                 N right singular vectors to be computed in VT
					if lwork >= n*n+3*n {
						//                    Sufficient workspace for a fast algorithm
						iu = 1
						if lwork >= wrkbl+a.Rows*n {
							//                       WORK(IU) is LDA by N
							ldwrku = a.Rows
						} else {
							//                       WORK(IU) is N by N
							ldwrku = n
						}
						itau = iu + ldwrku*n
						iwork = itau + n

						//                    Compute A=Q*R
						//                    (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy R to WORK(IU), zeroing out below it
						Zlacpy(Upper, n, n, a, work.Off(iu-1).CMatrix(ldwrku, opts))
						Zlaset(Lower, n-1, n-1, czero, czero, work.Off(iu).CMatrix(ldwrku, opts))

						//                    Generate Q in A
						//                    (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zungqr(m, n, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = 1
						itauq = itau
						itaup = itauq + n
						iwork = itaup + n

						//                    Bidiagonalize R in WORK(IU), copying result to VT
						//                    (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
						//                    (RWorkspace: need N)
						if err = Zgebrd(n, n, work.Off(iu-1).CMatrix(ldwrku, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Upper, n, n, work.Off(iu-1).CMatrix(ldwrku, opts), vt)

						//                    Generate left bidiagonalizing vectors in WORK(IU)
						//                    (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('Q', n, n, n, work.Off(iu-1).CMatrix(ldwrku, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right bidiagonalizing vectors in VT
						//                    (CWorkspace: need   N*N+3*N-1,
						//                                 prefer N*N+2*N+(N-1)*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('P', n, n, n, vt, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IU) and computing
						//                    right singular vectors of R in VT
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, n, n, n, 0, s, rwork.Off(ie-1), vt, work.Off(iu-1).CMatrix(ldwrku, opts), cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

						//                    Multiply Q in A by left singular vectors of R in
						//                    WORK(IU), storing result in U
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: 0)
						if err = u.Gemm(NoTrans, NoTrans, m, n, n, cone, a, work.Off(iu-1).CMatrix(ldwrku, opts), czero); err != nil {
							panic(err)
						}

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + n

						//                    Compute A=Q*R, copying result to U
						//                    (CWorkspace: need 2*N, prefer N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Lower, m, n, a, u)

						//                    Generate Q in U
						//                    (CWorkspace: need 2*N, prefer N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zungqr(m, n, n, u, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy R to VT, zeroing out below it
						Zlacpy(Upper, n, n, a, vt)
						if n > 1 {
							Zlaset(Lower, n-1, n-1, czero, czero, vt.Off(1, 0))
						}
						ie = 1
						itauq = itau
						itaup = itauq + n
						iwork = itaup + n

						//                    Bidiagonalize R in VT
						//                    (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
						//                    (RWorkspace: need N)
						if err = Zgebrd(n, n, vt, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply Q in U by left bidiagonalizing vectors
						//                    in VT
						//                    (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
						//                    (RWorkspace: 0)
						if err = Zunmbr('Q', Right, NoTrans, m, n, n, vt, work.Off(itauq-1), u, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right bidiagonalizing vectors in VT
						//                    (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('P', n, n, n, vt, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in VT
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, n, n, m, 0, s, rwork.Off(ie-1), vt, u, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

					}

				}

			} else if wntua {

				if wntvn {
					//                 Path 7 (M much larger than N, JOBU='A', JOBVT='N')
					//                 M left singular vectors to be computed in U and
					//                 no right singular vectors to be computed
					if lwork >= n*n+max(n+m, 3*n) {
						//                    Sufficient workspace for a fast algorithm
						ir = 1
						if lwork >= wrkbl+a.Rows*n {
							//                       WORK(IR) is LDA by N
							ldwrkr = a.Rows
						} else {
							//                       WORK(IR) is N by N
							ldwrkr = n
						}
						itau = ir + ldwrkr*n
						iwork = itau + n

						//                    Compute A=Q*R, copying result to U
						//                    (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Lower, m, n, a, u)

						//                    Copy R to WORK(IR), zeroing out below it
						Zlacpy(Upper, n, n, a, work.Off(ir-1).CMatrix(ldwrkr, opts))
						Zlaset(Lower, n-1, n-1, czero, czero, work.Off(ir).CMatrix(ldwrkr, opts))

						//                    Generate Q in U
						//                    (CWorkspace: need N*N+N+M, prefer N*N+N+M*NB)
						//                    (RWorkspace: 0)
						if err = Zungqr(m, m, n, u, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = 1
						itauq = itau
						itaup = itauq + n
						iwork = itaup + n

						//                    Bidiagonalize R in WORK(IR)
						//                    (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
						//                    (RWorkspace: need N)
						if err = Zgebrd(n, n, work.Off(ir-1).CMatrix(ldwrkr, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left bidiagonalizing vectors in WORK(IR)
						//                    (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('Q', n, n, n, work.Off(ir-1).CMatrix(ldwrkr, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IR)
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, n, 0, n, 0, s, rwork.Off(ie-1), cdum.CMatrix(1, opts), work.Off(ir-1).CMatrix(ldwrkr, opts), cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

						//                    Multiply Q in U by left singular vectors of R in
						//                    WORK(IR), storing result in A
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: 0)
						if err = a.Gemm(NoTrans, NoTrans, m, n, n, cone, u, work.Off(ir-1).CMatrix(ldwrkr, opts), czero); err != nil {
							panic(err)
						}

						//                    Copy left singular vectors of A from A to U
						Zlacpy(Full, m, n, a, u)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + n

						//                    Compute A=Q*R, copying result to U
						//                    (CWorkspace: need 2*N, prefer N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Lower, m, n, a, u)

						//                    Generate Q in U
						//                    (CWorkspace: need N+M, prefer N+M*NB)
						//                    (RWorkspace: 0)
						if err = Zungqr(m, m, n, u, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = 1
						itauq = itau
						itaup = itauq + n
						iwork = itaup + n

						//                    Zero out below R in A
						if n > 1 {
							Zlaset(Lower, n-1, n-1, czero, czero, a.Off(1, 0))
						}

						//                    Bidiagonalize R in A
						//                    (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
						//                    (RWorkspace: need N)
						if err = Zgebrd(n, n, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply Q in U by left bidiagonalizing vectors
						//                    in A
						//                    (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
						//                    (RWorkspace: 0)
						if err = Zunmbr('Q', Right, NoTrans, m, n, n, a, work.Off(itauq-1), u, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, n, 0, m, 0, s, rwork.Off(ie-1), cdum.CMatrix(1, opts), u, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

					}

				} else if wntvo {
					//                 Path 8 (M much larger than N, JOBU='A', JOBVT='O')
					//                 M left singular vectors to be computed in U and
					//                 N right singular vectors to be overwritten on A
					if lwork >= 2*n*n+max(n+m, 3*n) {
						//                    Sufficient workspace for a fast algorithm
						iu = 1
						if lwork >= wrkbl+2*a.Rows*n {
							//                       WORK(IU) is LDA by N and WORK(IR) is LDA by N
							ldwrku = a.Rows
							ir = iu + ldwrku*n
							ldwrkr = a.Rows
						} else if lwork >= wrkbl+(a.Rows+n)*n {
							//                       WORK(IU) is LDA by N and WORK(IR) is N by N
							ldwrku = a.Rows
							ir = iu + ldwrku*n
							ldwrkr = n
						} else {
							//                       WORK(IU) is N by N and WORK(IR) is N by N
							ldwrku = n
							ir = iu + ldwrku*n
							ldwrkr = n
						}
						itau = ir + ldwrkr*n
						iwork = itau + n

						//                    Compute A=Q*R, copying result to U
						//                    (CWorkspace: need 2*N*N+2*N, prefer 2*N*N+N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Lower, m, n, a, u)

						//                    Generate Q in U
						//                    (CWorkspace: need 2*N*N+N+M, prefer 2*N*N+N+M*NB)
						//                    (RWorkspace: 0)
						if err = Zungqr(m, m, n, u, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy R to WORK(IU), zeroing out below it
						Zlacpy(Upper, n, n, a, work.Off(iu-1).CMatrix(ldwrku, opts))
						Zlaset(Lower, n-1, n-1, czero, czero, work.Off(iu).CMatrix(ldwrku, opts))
						ie = 1
						itauq = itau
						itaup = itauq + n
						iwork = itaup + n

						//                    Bidiagonalize R in WORK(IU), copying result to
						//                    WORK(IR)
						//                    (CWorkspace: need   2*N*N+3*N,
						//                                 prefer 2*N*N+2*N+2*N*NB)
						//                    (RWorkspace: need   N)
						if err = Zgebrd(n, n, work.Off(iu-1).CMatrix(ldwrku, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Upper, n, n, work.Off(iu-1).CMatrix(ldwrku, opts), work.Off(ir-1).CMatrix(ldwrkr, opts))

						//                    Generate left bidiagonalizing vectors in WORK(IU)
						//                    (CWorkspace: need 2*N*N+3*N, prefer 2*N*N+2*N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('Q', n, n, n, work.Off(iu-1).CMatrix(ldwrku, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right bidiagonalizing vectors in WORK(IR)
						//                    (CWorkspace: need   2*N*N+3*N-1,
						//                                 prefer 2*N*N+2*N+(N-1)*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('P', n, n, n, work.Off(ir-1).CMatrix(ldwrkr, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IU) and computing
						//                    right singular vectors of R in WORK(IR)
						//                    (CWorkspace: need 2*N*N)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, n, n, n, 0, s, rwork.Off(ie-1), work.Off(ir-1).CMatrix(ldwrkr, opts), work.Off(iu-1).CMatrix(ldwrku, opts), cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

						//                    Multiply Q in U by left singular vectors of R in
						//                    WORK(IU), storing result in A
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: 0)
						if err = a.Gemm(NoTrans, NoTrans, m, n, n, cone, u, work.Off(iu-1).CMatrix(ldwrku, opts), czero); err != nil {
							panic(err)
						}

						//                    Copy left singular vectors of A from A to U
						Zlacpy(Full, m, n, a, u)

						//                    Copy right singular vectors of R from WORK(IR) to A
						Zlacpy(Full, n, n, work.Off(ir-1).CMatrix(ldwrkr, opts), a)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + n

						//                    Compute A=Q*R, copying result to U
						//                    (CWorkspace: need 2*N, prefer N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Lower, m, n, a, u)

						//                    Generate Q in U
						//                    (CWorkspace: need N+M, prefer N+M*NB)
						//                    (RWorkspace: 0)
						if err = Zungqr(m, m, n, u, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = 1
						itauq = itau
						itaup = itauq + n
						iwork = itaup + n

						//                    Zero out below R in A
						if n > 1 {
							Zlaset(Lower, n-1, n-1, czero, czero, a.Off(1, 0))
						}

						//                    Bidiagonalize R in A
						//                    (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
						//                    (RWorkspace: need N)
						if err = Zgebrd(n, n, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply Q in U by left bidiagonalizing vectors
						//                    in A
						//                    (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
						//                    (RWorkspace: 0)
						if err = Zunmbr('Q', Right, NoTrans, m, n, n, a, work.Off(itauq-1), u, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right bidiagonalizing vectors in A
						//                    (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('P', n, n, n, a, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in A
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, n, n, m, 0, s, rwork.Off(ie-1), a, u, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

					}

				} else if wntvas {
					//                 Path 9 (M much larger than N, JOBU='A', JOBVT='S'
					//                         or 'A')
					//                 M left singular vectors to be computed in U and
					//                 N right singular vectors to be computed in VT
					if lwork >= n*n+max(n+m, 3*n) {
						//                    Sufficient workspace for a fast algorithm
						iu = 1
						if lwork >= wrkbl+a.Rows*n {
							//                       WORK(IU) is LDA by N
							ldwrku = a.Rows
						} else {
							//                       WORK(IU) is N by N
							ldwrku = n
						}
						itau = iu + ldwrku*n
						iwork = itau + n

						//                    Compute A=Q*R, copying result to U
						//                    (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Lower, m, n, a, u)

						//                    Generate Q in U
						//                    (CWorkspace: need N*N+N+M, prefer N*N+N+M*NB)
						//                    (RWorkspace: 0)
						if err = Zungqr(m, m, n, u, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy R to WORK(IU), zeroing out below it
						Zlacpy(Upper, n, n, a, work.Off(iu-1).CMatrix(ldwrku, opts))
						Zlaset(Lower, n-1, n-1, czero, czero, work.Off(iu).CMatrix(ldwrku, opts))
						ie = 1
						itauq = itau
						itaup = itauq + n
						iwork = itaup + n

						//                    Bidiagonalize R in WORK(IU), copying result to VT
						//                    (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
						//                    (RWorkspace: need N)
						if err = Zgebrd(n, n, work.Off(iu-1).CMatrix(ldwrku, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Upper, n, n, work.Off(iu-1).CMatrix(ldwrku, opts), vt)

						//                    Generate left bidiagonalizing vectors in WORK(IU)
						//                    (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('Q', n, n, n, work.Off(iu-1).CMatrix(ldwrku, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right bidiagonalizing vectors in VT
						//                    (CWorkspace: need   N*N+3*N-1,
						//                                 prefer N*N+2*N+(N-1)*NB)
						//                    (RWorkspace: need   0)
						if err = Zungbr('P', n, n, n, vt, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IU) and computing
						//                    right singular vectors of R in VT
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, n, n, n, 0, s, rwork.Off(ie-1), vt, work.Off(iu-1).CMatrix(ldwrku, opts), cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

						//                    Multiply Q in U by left singular vectors of R in
						//                    WORK(IU), storing result in A
						//                    (CWorkspace: need N*N)
						//                    (RWorkspace: 0)
						if err = a.Gemm(NoTrans, NoTrans, m, n, n, cone, u, work.Off(iu-1).CMatrix(ldwrku, opts), czero); err != nil {
							panic(err)
						}

						//                    Copy left singular vectors of A from A to U
						Zlacpy(Full, m, n, a, u)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + n

						//                    Compute A=Q*R, copying result to U
						//                    (CWorkspace: need 2*N, prefer N+N*NB)
						//                    (RWorkspace: 0)
						if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Lower, m, n, a, u)

						//                    Generate Q in U
						//                    (CWorkspace: need N+M, prefer N+M*NB)
						//                    (RWorkspace: 0)
						if err = Zungqr(m, m, n, u, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy R from A to VT, zeroing out below it
						Zlacpy(Upper, n, n, a, vt)
						if n > 1 {
							Zlaset(Lower, n-1, n-1, czero, czero, vt.Off(1, 0))
						}
						ie = 1
						itauq = itau
						itaup = itauq + n
						iwork = itaup + n

						//                    Bidiagonalize R in VT
						//                    (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
						//                    (RWorkspace: need N)
						if err = Zgebrd(n, n, vt, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply Q in U by left bidiagonalizing vectors
						//                    in VT
						//                    (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
						//                    (RWorkspace: 0)
						if err = Zunmbr('Q', Right, NoTrans, m, n, n, vt, work.Off(itauq-1), u, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right bidiagonalizing vectors in VT
						//                    (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('P', n, n, n, vt, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in VT
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, n, n, m, 0, s, rwork.Off(ie-1), vt, u, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

					}

				}

			}

		} else {
			//           M .LT. MNTHR
			//
			//           Path 10 (M at least N, but not much larger)
			//           Reduce to bidiagonal form without QR decomposition
			ie = 1
			itauq = 1
			itaup = itauq + n
			iwork = itaup + n

			//           Bidiagonalize A
			//           (CWorkspace: need 2*N+M, prefer 2*N+(M+N)*NB)
			//           (RWorkspace: need N)
			if err = Zgebrd(m, n, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
				panic(err)
			}
			if wntuas {
				//              If left singular vectors desired in U, copy result to U
				//              and generate left bidiagonalizing vectors in U
				//              (CWorkspace: need 2*N+NCU, prefer 2*N+NCU*NB)
				//              (RWorkspace: 0)
				Zlacpy(Lower, m, n, a, u)
				if wntus {
					ncu = n
				}
				if wntua {
					ncu = m
				}
				if err = Zungbr('Q', m, ncu, n, u, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
			}
			if wntvas {
				//              If right singular vectors desired in VT, copy result to
				//              VT and generate right bidiagonalizing vectors in VT
				//              (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
				//              (RWorkspace: 0)
				Zlacpy(Upper, n, n, a, vt)
				if err = Zungbr('P', n, n, n, vt, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
			}
			if wntuo {
				//              If left singular vectors desired in A, generate left
				//              bidiagonalizing vectors in A
				//              (CWorkspace: need 3*N, prefer 2*N+N*NB)
				//              (RWorkspace: 0)
				if err = Zungbr('Q', m, n, n, a, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
			}
			if wntvo {
				//              If right singular vectors desired in A, generate right
				//              bidiagonalizing vectors in A
				//              (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
				//              (RWorkspace: 0)
				if err = Zungbr('P', n, n, n, a, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
			}
			irwork = ie + n
			if wntuas || wntuo {
				nru = m
			}
			if wntun {
				nru = 0
			}
			if wntvas || wntvo {
				ncvt = n
			}
			if wntvn {
				ncvt = 0
			}
			if (!wntuo) && (!wntvo) {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in U and computing right singular
				//              vectors in VT
				//              (CWorkspace: 0)
				//              (RWorkspace: need BDSPAC)
				if info, err = Zbdsqr(Upper, n, ncvt, nru, 0, s, rwork.Off(ie-1), vt, u, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
					panic(err)
				}
			} else if (!wntuo) && wntvo {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in U and computing right singular
				//              vectors in A
				//              (CWorkspace: 0)
				//              (RWorkspace: need BDSPAC)
				if info, err = Zbdsqr(Upper, n, ncvt, nru, 0, s, rwork.Off(ie-1), a, u, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
					panic(err)
				}
			} else {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in A and computing right singular
				//              vectors in VT
				//              (CWorkspace: 0)
				//              (RWorkspace: need BDSPAC)
				if info, err = Zbdsqr(Upper, n, ncvt, nru, 0, s, rwork.Off(ie-1), vt, a, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
					panic(err)
				}
			}

		}

	} else {
		//        A has more columns than rows. If A has sufficiently more
		//        columns than rows, first reduce using the LQ decomposition (if
		//        sufficient workspace available)
		if n >= mnthr {

			if wntvn {
				//              Path 1t(N much larger than M, JOBVT='N')
				//              No right singular vectors to be computed
				itau = 1
				iwork = itau + m

				//              Compute A=L*Q
				//              (CWorkspace: need 2*M, prefer M+M*NB)
				//              (RWorkspace: 0)
				if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}

				//              Zero out above L
				Zlaset(Upper, m-1, m-1, czero, czero, a.Off(0, 1))
				ie = 1
				itauq = 1
				itaup = itauq + m
				iwork = itaup + m

				//              Bidiagonalize L in A
				//              (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
				//              (RWorkspace: need M)
				if err = Zgebrd(m, m, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
				if wntuo || wntuas {
					//                 If left singular vectors desired, generate Q
					//                 (CWorkspace: need 3*M, prefer 2*M+M*NB)
					//                 (RWorkspace: 0)
					if err = Zungbr('Q', m, m, m, a, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
				}
				irwork = ie + m
				nru = 0
				if wntuo || wntuas {
					nru = m
				}

				//              Perform bidiagonal QR iteration, computing left singular
				//              vectors of A in A if desired
				//              (CWorkspace: 0)
				//              (RWorkspace: need BDSPAC)
				if info, err = Zbdsqr(Upper, m, 0, nru, 0, s, rwork.Off(ie-1), cdum.CMatrix(1, opts), a, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
					panic(err)
				}

				//              If left singular vectors desired in U, copy them there
				if wntuas {
					Zlacpy(Full, m, m, a, u)
				}

			} else if wntvo && wntun {
				//              Path 2t(N much larger than M, JOBU='N', JOBVT='O')
				//              M right singular vectors to be overwritten on A and
				//              no left singular vectors to be computed
				if lwork >= m*m+3*m {
					//                 Sufficient workspace for a fast algorithm
					ir = 1
					if lwork >= max(wrkbl, a.Rows*n)+a.Rows*m {
						//                    WORK(IU) is LDA by N and WORK(IR) is LDA by M
						ldwrku = a.Rows
						chunk = n
						ldwrkr = a.Rows
					} else if lwork >= max(wrkbl, a.Rows*n)+m*m {
						//                    WORK(IU) is LDA by N and WORK(IR) is M by M
						ldwrku = a.Rows
						chunk = n
						ldwrkr = m
					} else {
						//                    WORK(IU) is M by CHUNK and WORK(IR) is M by M
						ldwrku = m
						chunk = (lwork - m*m) / m
						ldwrkr = m
					}
					itau = ir + ldwrkr*m
					iwork = itau + m

					//                 Compute A=L*Q
					//                 (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
					//                 (RWorkspace: 0)
					if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Copy L to WORK(IR) and zero out above it
					Zlacpy(Lower, m, m, a, work.Off(ir-1).CMatrix(ldwrkr, opts))
					Zlaset(Upper, m-1, m-1, czero, czero, work.Off(ir+ldwrkr-1).CMatrix(ldwrkr, opts))

					//                 Generate Q in A
					//                 (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
					//                 (RWorkspace: 0)
					if err = Zunglq(m, n, m, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					ie = 1
					itauq = itau
					itaup = itauq + m
					iwork = itaup + m

					//                 Bidiagonalize L in WORK(IR)
					//                 (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
					//                 (RWorkspace: need M)
					if err = Zgebrd(m, m, work.Off(ir-1).CMatrix(ldwrkr, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Generate right vectors bidiagonalizing L
					//                 (CWorkspace: need M*M+3*M-1, prefer M*M+2*M+(M-1)*NB)
					//                 (RWorkspace: 0)
					if err = Zungbr('P', m, m, m, work.Off(ir-1).CMatrix(ldwrkr, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					irwork = ie + m

					//                 Perform bidiagonal QR iteration, computing right
					//                 singular vectors of L in WORK(IR)
					//                 (CWorkspace: need M*M)
					//                 (RWorkspace: need BDSPAC)
					if info, err = Zbdsqr(Upper, m, m, 0, 0, s, rwork.Off(ie-1), work.Off(ir-1).CMatrix(ldwrkr, opts), cdum.CMatrix(1, opts), cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
						panic(err)
					}
					iu = itauq

					//                 Multiply right singular vectors of L in WORK(IR) by Q
					//                 in A, storing result in WORK(IU) and copying to A
					//                 (CWorkspace: need M*M+M, prefer M*M+M*N)
					//                 (RWorkspace: 0)
					for i = 1; i <= n; i += chunk {
						blk = min(n-i+1, chunk)
						if err = work.Off(iu-1).CMatrix(ldwrku, opts).Gemm(NoTrans, NoTrans, m, blk, m, cone, work.Off(ir-1).CMatrix(ldwrkr, opts), a.Off(0, i-1), czero); err != nil {
							panic(err)
						}
						Zlacpy(Full, m, blk, work.Off(iu-1).CMatrix(ldwrku, opts), a.Off(0, i-1))
					}

				} else {

					//                 Insufficient workspace for a fast algorithm
					ie = 1
					itauq = 1
					itaup = itauq + m
					iwork = itaup + m

					//                 Bidiagonalize A
					//                 (CWorkspace: need 2*M+N, prefer 2*M+(M+N)*NB)
					//                 (RWorkspace: need M)
					if err = Zgebrd(m, n, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Generate right vectors bidiagonalizing A
					//                 (CWorkspace: need 3*M, prefer 2*M+M*NB)
					//                 (RWorkspace: 0)
					if err = Zungbr('P', m, n, m, a, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					irwork = ie + m

					//                 Perform bidiagonal QR iteration, computing right
					//                 singular vectors of A in A
					//                 (CWorkspace: 0)
					//                 (RWorkspace: need BDSPAC)
					if info, err = Zbdsqr(Lower, m, n, 0, 0, s, rwork.Off(ie-1), a, cdum.CMatrix(1, opts), cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
						panic(err)
					}

				}

			} else if wntvo && wntuas {
				//              Path 3t(N much larger than M, JOBU='S' or 'A', JOBVT='O')
				//              M right singular vectors to be overwritten on A and
				//              M left singular vectors to be computed in U
				if lwork >= m*m+3*m {
					//                 Sufficient workspace for a fast algorithm
					ir = 1
					if lwork >= max(wrkbl, a.Rows*n)+a.Rows*m {
						//                    WORK(IU) is LDA by N and WORK(IR) is LDA by M
						ldwrku = a.Rows
						chunk = n
						ldwrkr = a.Rows
					} else if lwork >= max(wrkbl, a.Rows*n)+m*m {
						//                    WORK(IU) is LDA by N and WORK(IR) is M by M
						ldwrku = a.Rows
						chunk = n
						ldwrkr = m
					} else {
						//                    WORK(IU) is M by CHUNK and WORK(IR) is M by M
						ldwrku = m
						chunk = (lwork - m*m) / m
						ldwrkr = m
					}
					itau = ir + ldwrkr*m
					iwork = itau + m

					//                 Compute A=L*Q
					//                 (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
					//                 (RWorkspace: 0)
					if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Copy L to U, zeroing about above it
					Zlacpy(Lower, m, m, a, u)
					Zlaset(Upper, m-1, m-1, czero, czero, u.Off(0, 1))

					//                 Generate Q in A
					//                 (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
					//                 (RWorkspace: 0)
					if err = Zunglq(m, n, m, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					ie = 1
					itauq = itau
					itaup = itauq + m
					iwork = itaup + m

					//                 Bidiagonalize L in U, copying result to WORK(IR)
					//                 (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
					//                 (RWorkspace: need M)
					if err = Zgebrd(m, m, u, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					Zlacpy(Upper, m, m, u, work.Off(ir-1).CMatrix(ldwrkr, opts))

					//                 Generate right vectors bidiagonalizing L in WORK(IR)
					//                 (CWorkspace: need M*M+3*M-1, prefer M*M+2*M+(M-1)*NB)
					//                 (RWorkspace: 0)
					if err = Zungbr('P', m, m, m, work.Off(ir-1).CMatrix(ldwrkr, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Generate left vectors bidiagonalizing L in U
					//                 (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB)
					//                 (RWorkspace: 0)
					if err = Zungbr('Q', m, m, m, u, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					irwork = ie + m

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of L in U, and computing right
					//                 singular vectors of L in WORK(IR)
					//                 (CWorkspace: need M*M)
					//                 (RWorkspace: need BDSPAC)
					if info, err = Zbdsqr(Upper, m, m, m, 0, s, rwork.Off(ie-1), work.Off(ir-1).CMatrix(ldwrkr, opts), u, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
						panic(err)
					}
					iu = itauq

					//                 Multiply right singular vectors of L in WORK(IR) by Q
					//                 in A, storing result in WORK(IU) and copying to A
					//                 (CWorkspace: need M*M+M, prefer M*M+M*N))
					//                 (RWorkspace: 0)
					for i = 1; i <= n; i += chunk {
						blk = min(n-i+1, chunk)
						if err = work.Off(iu-1).CMatrix(ldwrku, opts).Gemm(NoTrans, NoTrans, m, blk, m, cone, work.Off(ir-1).CMatrix(ldwrkr, opts), a.Off(0, i-1), czero); err != nil {
							panic(err)
						}
						Zlacpy(Full, m, blk, work.Off(iu-1).CMatrix(ldwrku, opts), a.Off(0, i-1))
					}

				} else {

					//                 Insufficient workspace for a fast algorithm
					itau = 1
					iwork = itau + m

					//                 Compute A=L*Q
					//                 (CWorkspace: need 2*M, prefer M+M*NB)
					//                 (RWorkspace: 0)
					if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Copy L to U, zeroing out above it
					Zlacpy(Lower, m, m, a, u)
					Zlaset(Upper, m-1, m-1, czero, czero, u.Off(0, 1))

					//                 Generate Q in A
					//                 (CWorkspace: need 2*M, prefer M+M*NB)
					//                 (RWorkspace: 0)
					if err = Zunglq(m, n, m, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					ie = 1
					itauq = itau
					itaup = itauq + m
					iwork = itaup + m

					//                 Bidiagonalize L in U
					//                 (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
					//                 (RWorkspace: need M)
					if err = Zgebrd(m, m, u, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Multiply right vectors bidiagonalizing L by Q in A
					//                 (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
					//                 (RWorkspace: 0)
					if err = Zunmbr('P', Left, ConjTrans, m, n, m, u, work.Off(itaup-1), a, work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Generate left vectors bidiagonalizing L in U
					//                 (CWorkspace: need 3*M, prefer 2*M+M*NB)
					//                 (RWorkspace: 0)
					if err = Zungbr('Q', m, m, m, u, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					irwork = ie + m

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of A in U and computing right
					//                 singular vectors of A in A
					//                 (CWorkspace: 0)
					//                 (RWorkspace: need BDSPAC)
					if info, err = Zbdsqr(Upper, m, n, m, 0, s, rwork.Off(ie-1), a, u, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
						panic(err)
					}

				}

			} else if wntvs {

				if wntun {
					//                 Path 4t(N much larger than M, JOBU='N', JOBVT='S')
					//                 M right singular vectors to be computed in VT and
					//                 no left singular vectors to be computed
					if lwork >= m*m+3*m {
						//                    Sufficient workspace for a fast algorithm
						ir = 1
						if lwork >= wrkbl+a.Rows*m {
							//                       WORK(IR) is LDA by M
							ldwrkr = a.Rows
						} else {
							//                       WORK(IR) is M by M
							ldwrkr = m
						}
						itau = ir + ldwrkr*m
						iwork = itau + m

						//                    Compute A=L*Q
						//                    (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy L to WORK(IR), zeroing out above it
						Zlacpy(Lower, m, m, a, work.Off(ir-1).CMatrix(ldwrkr, opts))
						Zlaset(Upper, m-1, m-1, czero, czero, work.Off(ir+ldwrkr-1).CMatrix(ldwrkr, opts))

						//                    Generate Q in A
						//                    (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zunglq(m, n, m, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = 1
						itauq = itau
						itaup = itauq + m
						iwork = itaup + m

						//                    Bidiagonalize L in WORK(IR)
						//                    (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
						//                    (RWorkspace: need M)
						if err = Zgebrd(m, m, work.Off(ir-1).CMatrix(ldwrkr, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right vectors bidiagonalizing L in
						//                    WORK(IR)
						//                    (CWorkspace: need M*M+3*M, prefer M*M+2*M+(M-1)*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('P', m, m, m, work.Off(ir-1).CMatrix(ldwrkr, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + m

						//                    Perform bidiagonal QR iteration, computing right
						//                    singular vectors of L in WORK(IR)
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, m, m, 0, 0, s, rwork.Off(ie-1), work.Off(ir-1).CMatrix(ldwrkr, opts), cdum.CMatrix(1, opts), cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

						//                    Multiply right singular vectors of L in WORK(IR) by
						//                    Q in A, storing result in VT
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: 0)
						if err = vt.Gemm(NoTrans, NoTrans, m, n, m, cone, work.Off(ir-1).CMatrix(ldwrkr, opts), a, czero); err != nil {
							panic(err)
						}

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + m

						//                    Compute A=L*Q
						//                    (CWorkspace: need 2*M, prefer M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy result to VT
						Zlacpy(Upper, m, n, a, vt)

						//                    Generate Q in VT
						//                    (CWorkspace: need 2*M, prefer M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zunglq(m, n, m, vt, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = 1
						itauq = itau
						itaup = itauq + m
						iwork = itaup + m

						//                    Zero out above L in A
						Zlaset(Upper, m-1, m-1, czero, czero, a.Off(0, 1))

						//                    Bidiagonalize L in A
						//                    (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
						//                    (RWorkspace: need M)
						if err = Zgebrd(m, m, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply right vectors bidiagonalizing L by Q in VT
						//                    (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
						//                    (RWorkspace: 0)
						if err = Zunmbr('P', Left, ConjTrans, m, n, m, a, work.Off(itaup-1), vt, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + m

						//                    Perform bidiagonal QR iteration, computing right
						//                    singular vectors of A in VT
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, m, n, 0, 0, s, rwork.Off(ie-1), vt, cdum.CMatrix(1, opts), cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

					}

				} else if wntuo {
					//                 Path 5t(N much larger than M, JOBU='O', JOBVT='S')
					//                 M right singular vectors to be computed in VT and
					//                 M left singular vectors to be overwritten on A
					if lwork >= 2*m*m+3*m {
						//                    Sufficient workspace for a fast algorithm
						iu = 1
						if lwork >= wrkbl+2*a.Rows*m {
							//                       WORK(IU) is LDA by M and WORK(IR) is LDA by M
							ldwrku = a.Rows
							ir = iu + ldwrku*m
							ldwrkr = a.Rows
						} else if lwork >= wrkbl+(a.Rows+m)*m {
							//                       WORK(IU) is LDA by M and WORK(IR) is M by M
							ldwrku = a.Rows
							ir = iu + ldwrku*m
							ldwrkr = m
						} else {
							//                       WORK(IU) is M by M and WORK(IR) is M by M
							ldwrku = m
							ir = iu + ldwrku*m
							ldwrkr = m
						}
						itau = ir + ldwrkr*m
						iwork = itau + m

						//                    Compute A=L*Q
						//                    (CWorkspace: need 2*M*M+2*M, prefer 2*M*M+M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy L to WORK(IU), zeroing out below it
						Zlacpy(Lower, m, m, a, work.Off(iu-1).CMatrix(ldwrku, opts))
						Zlaset(Upper, m-1, m-1, czero, czero, work.Off(iu+ldwrku-1).CMatrix(ldwrku, opts))

						//                    Generate Q in A
						//                    (CWorkspace: need 2*M*M+2*M, prefer 2*M*M+M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zunglq(m, n, m, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = 1
						itauq = itau
						itaup = itauq + m
						iwork = itaup + m

						//                    Bidiagonalize L in WORK(IU), copying result to
						//                    WORK(IR)
						//                    (CWorkspace: need   2*M*M+3*M,
						//                                 prefer 2*M*M+2*M+2*M*NB)
						//                    (RWorkspace: need   M)
						if err = Zgebrd(m, m, work.Off(iu-1).CMatrix(ldwrku, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Lower, m, m, work.Off(iu-1).CMatrix(ldwrku, opts), work.Off(ir-1).CMatrix(ldwrkr, opts))

						//                    Generate right bidiagonalizing vectors in WORK(IU)
						//                    (CWorkspace: need   2*M*M+3*M-1,
						//                                 prefer 2*M*M+2*M+(M-1)*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('P', m, m, m, work.Off(iu-1).CMatrix(ldwrku, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left bidiagonalizing vectors in WORK(IR)
						//                    (CWorkspace: need 2*M*M+3*M, prefer 2*M*M+2*M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('Q', m, m, m, work.Off(ir-1).CMatrix(ldwrkr, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + m

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of L in WORK(IR) and computing
						//                    right singular vectors of L in WORK(IU)
						//                    (CWorkspace: need 2*M*M)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, m, m, m, 0, s, rwork.Off(ie-1), work.Off(iu-1).CMatrix(ldwrku, opts), work.Off(ir-1).CMatrix(ldwrkr, opts), cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

						//                    Multiply right singular vectors of L in WORK(IU) by
						//                    Q in A, storing result in VT
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: 0)
						if err = vt.Gemm(NoTrans, NoTrans, m, n, m, cone, work.Off(iu-1).CMatrix(ldwrku, opts), a, czero); err != nil {
							panic(err)
						}

						//                    Copy left singular vectors of L to A
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: 0)
						Zlacpy(Full, m, m, work.Off(ir-1).CMatrix(ldwrkr, opts), a)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + m

						//                    Compute A=L*Q, copying result to VT
						//                    (CWorkspace: need 2*M, prefer M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Upper, m, n, a, vt)

						//                    Generate Q in VT
						//                    (CWorkspace: need 2*M, prefer M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zunglq(m, n, m, vt, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = 1
						itauq = itau
						itaup = itauq + m
						iwork = itaup + m

						//                    Zero out above L in A
						Zlaset(Upper, m-1, m-1, czero, czero, a.Off(0, 1))

						//                    Bidiagonalize L in A
						//                    (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
						//                    (RWorkspace: need M)
						if err = Zgebrd(m, m, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply right vectors bidiagonalizing L by Q in VT
						//                    (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
						//                    (RWorkspace: 0)
						if err = Zunmbr('P', Left, ConjTrans, m, n, m, a, work.Off(itaup-1), vt, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left bidiagonalizing vectors of L in A
						//                    (CWorkspace: need 3*M, prefer 2*M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('Q', m, m, m, a, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + m

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in A and computing right
						//                    singular vectors of A in VT
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, m, n, m, 0, s, rwork.Off(ie-1), vt, a, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

					}

				} else if wntuas {
					//                 Path 6t(N much larger than M, JOBU='S' or 'A',
					//                         JOBVT='S')
					//                 M right singular vectors to be computed in VT and
					//                 M left singular vectors to be computed in U
					if lwork >= m*m+3*m {
						//                    Sufficient workspace for a fast algorithm
						iu = 1
						if lwork >= wrkbl+a.Rows*m {
							//                       WORK(IU) is LDA by N
							ldwrku = a.Rows
						} else {
							//                       WORK(IU) is LDA by M
							ldwrku = m
						}
						itau = iu + ldwrku*m
						iwork = itau + m

						//                    Compute A=L*Q
						//                    (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy L to WORK(IU), zeroing out above it
						Zlacpy(Lower, m, m, a, work.Off(iu-1).CMatrix(ldwrku, opts))
						Zlaset(Upper, m-1, m-1, czero, czero, work.Off(iu+ldwrku-1).CMatrix(ldwrku, opts))

						//                    Generate Q in A
						//                    (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zunglq(m, n, m, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = 1
						itauq = itau
						itaup = itauq + m
						iwork = itaup + m

						//                    Bidiagonalize L in WORK(IU), copying result to U
						//                    (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
						//                    (RWorkspace: need M)
						if err = Zgebrd(m, m, work.Off(iu-1).CMatrix(ldwrku, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Lower, m, m, work.Off(iu-1).CMatrix(ldwrku, opts), u)

						//                    Generate right bidiagonalizing vectors in WORK(IU)
						//                    (CWorkspace: need   M*M+3*M-1,
						//                                 prefer M*M+2*M+(M-1)*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('P', m, m, m, work.Off(iu-1).CMatrix(ldwrku, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left bidiagonalizing vectors in U
						//                    (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('Q', m, m, m, u, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + m

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of L in U and computing right
						//                    singular vectors of L in WORK(IU)
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, m, m, m, 0, s, rwork.Off(ie-1), work.Off(iu-1).CMatrix(ldwrku, opts), u, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

						//                    Multiply right singular vectors of L in WORK(IU) by
						//                    Q in A, storing result in VT
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: 0)
						if err = vt.Gemm(NoTrans, NoTrans, m, n, m, cone, work.Off(iu-1).CMatrix(ldwrku, opts), a, czero); err != nil {
							panic(err)
						}

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + m

						//                    Compute A=L*Q, copying result to VT
						//                    (CWorkspace: need 2*M, prefer M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Upper, m, n, a, vt)

						//                    Generate Q in VT
						//                    (CWorkspace: need 2*M, prefer M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zunglq(m, n, m, vt, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy L to U, zeroing out above it
						Zlacpy(Lower, m, m, a, u)
						Zlaset(Upper, m-1, m-1, czero, czero, u.Off(0, 1))
						ie = 1
						itauq = itau
						itaup = itauq + m
						iwork = itaup + m

						//                    Bidiagonalize L in U
						//                    (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
						//                    (RWorkspace: need M)
						if err = Zgebrd(m, m, u, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply right bidiagonalizing vectors in U by Q
						//                    in VT
						//                    (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
						//                    (RWorkspace: 0)
						if err = Zunmbr('P', Left, ConjTrans, m, n, m, u, work.Off(itaup-1), vt, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left bidiagonalizing vectors in U
						//                    (CWorkspace: need 3*M, prefer 2*M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('Q', m, m, m, u, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + m

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in VT
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, m, n, m, 0, s, rwork.Off(ie-1), vt, u, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

					}

				}

			} else if wntva {

				if wntun {
					//                 Path 7t(N much larger than M, JOBU='N', JOBVT='A')
					//                 N right singular vectors to be computed in VT and
					//                 no left singular vectors to be computed
					if lwork >= m*m+max(n+m, 3*m) {
						//                    Sufficient workspace for a fast algorithm
						ir = 1
						if lwork >= wrkbl+a.Rows*m {
							//                       WORK(IR) is LDA by M
							ldwrkr = a.Rows
						} else {
							//                       WORK(IR) is M by M
							ldwrkr = m
						}
						itau = ir + ldwrkr*m
						iwork = itau + m

						//                    Compute A=L*Q, copying result to VT
						//                    (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Upper, m, n, a, vt)

						//                    Copy L to WORK(IR), zeroing out above it
						Zlacpy(Lower, m, m, a, work.Off(ir-1).CMatrix(ldwrkr, opts))
						Zlaset(Upper, m-1, m-1, czero, czero, work.Off(ir+ldwrkr-1).CMatrix(ldwrkr, opts))

						//                    Generate Q in VT
						//                    (CWorkspace: need M*M+M+N, prefer M*M+M+N*NB)
						//                    (RWorkspace: 0)
						if err = Zunglq(n, n, m, vt, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = 1
						itauq = itau
						itaup = itauq + m
						iwork = itaup + m

						//                    Bidiagonalize L in WORK(IR)
						//                    (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
						//                    (RWorkspace: need M)
						if err = Zgebrd(m, m, work.Off(ir-1).CMatrix(ldwrkr, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right bidiagonalizing vectors in WORK(IR)
						//                    (CWorkspace: need   M*M+3*M-1,
						//                                 prefer M*M+2*M+(M-1)*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('P', m, m, m, work.Off(ir-1).CMatrix(ldwrkr, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + m

						//                    Perform bidiagonal QR iteration, computing right
						//                    singular vectors of L in WORK(IR)
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, m, m, 0, 0, s, rwork.Off(ie-1), work.Off(ir-1).CMatrix(ldwrkr, opts), cdum.CMatrix(1, opts), cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

						//                    Multiply right singular vectors of L in WORK(IR) by
						//                    Q in VT, storing result in A
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: 0)
						if err = a.Gemm(NoTrans, NoTrans, m, n, m, cone, work.Off(ir-1).CMatrix(ldwrkr, opts), vt, czero); err != nil {
							panic(err)
						}

						//                    Copy right singular vectors of A from A to VT
						Zlacpy(Full, m, n, a, vt)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + m

						//                    Compute A=L*Q, copying result to VT
						//                    (CWorkspace: need 2*M, prefer M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Upper, m, n, a, vt)

						//                    Generate Q in VT
						//                    (CWorkspace: need M+N, prefer M+N*NB)
						//                    (RWorkspace: 0)
						if err = Zunglq(n, n, m, vt, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = 1
						itauq = itau
						itaup = itauq + m
						iwork = itaup + m

						//                    Zero out above L in A
						Zlaset(Upper, m-1, m-1, czero, czero, a.Off(0, 1))

						//                    Bidiagonalize L in A
						//                    (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
						//                    (RWorkspace: need M)
						if err = Zgebrd(m, m, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply right bidiagonalizing vectors in A by Q
						//                    in VT
						//                    (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
						//                    (RWorkspace: 0)
						if err = Zunmbr('P', Left, ConjTrans, m, n, m, a, work.Off(itaup-1), vt, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + m

						//                    Perform bidiagonal QR iteration, computing right
						//                    singular vectors of A in VT
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, m, n, 0, 0, s, rwork.Off(ie-1), vt, cdum.CMatrix(1, opts), cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

					}

				} else if wntuo {
					//                 Path 8t(N much larger than M, JOBU='O', JOBVT='A')
					//                 N right singular vectors to be computed in VT and
					//                 M left singular vectors to be overwritten on A
					if lwork >= 2*m*m+max(n+m, 3*m) {
						//                    Sufficient workspace for a fast algorithm
						iu = 1
						if lwork >= wrkbl+2*a.Rows*m {
							//                       WORK(IU) is LDA by M and WORK(IR) is LDA by M
							ldwrku = a.Rows
							ir = iu + ldwrku*m
							ldwrkr = a.Rows
						} else if lwork >= wrkbl+(a.Rows+m)*m {
							//                       WORK(IU) is LDA by M and WORK(IR) is M by M
							ldwrku = a.Rows
							ir = iu + ldwrku*m
							ldwrkr = m
						} else {
							//                       WORK(IU) is M by M and WORK(IR) is M by M
							ldwrku = m
							ir = iu + ldwrku*m
							ldwrkr = m
						}
						itau = ir + ldwrkr*m
						iwork = itau + m

						//                    Compute A=L*Q, copying result to VT
						//                    (CWorkspace: need 2*M*M+2*M, prefer 2*M*M+M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Upper, m, n, a, vt)

						//                    Generate Q in VT
						//                    (CWorkspace: need 2*M*M+M+N, prefer 2*M*M+M+N*NB)
						//                    (RWorkspace: 0)
						if err = Zunglq(n, n, m, vt, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy L to WORK(IU), zeroing out above it
						Zlacpy(Lower, m, m, a, work.Off(iu-1).CMatrix(ldwrku, opts))
						Zlaset(Upper, m-1, m-1, czero, czero, work.Off(iu+ldwrku-1).CMatrix(ldwrku, opts))
						ie = 1
						itauq = itau
						itaup = itauq + m
						iwork = itaup + m

						//                    Bidiagonalize L in WORK(IU), copying result to
						//                    WORK(IR)
						//                    (CWorkspace: need   2*M*M+3*M,
						//                                 prefer 2*M*M+2*M+2*M*NB)
						//                    (RWorkspace: need   M)
						if err = Zgebrd(m, m, work.Off(iu-1).CMatrix(ldwrku, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Lower, m, m, work.Off(iu-1).CMatrix(ldwrku, opts), work.Off(ir-1).CMatrix(ldwrkr, opts))

						//                    Generate right bidiagonalizing vectors in WORK(IU)
						//                    (CWorkspace: need   2*M*M+3*M-1,
						//                                 prefer 2*M*M+2*M+(M-1)*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('P', m, m, m, work.Off(iu-1).CMatrix(ldwrku, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left bidiagonalizing vectors in WORK(IR)
						//                    (CWorkspace: need 2*M*M+3*M, prefer 2*M*M+2*M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('Q', m, m, m, work.Off(ir-1).CMatrix(ldwrkr, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + m

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of L in WORK(IR) and computing
						//                    right singular vectors of L in WORK(IU)
						//                    (CWorkspace: need 2*M*M)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, m, m, m, 0, s, rwork.Off(ie-1), work.Off(iu-1).CMatrix(ldwrku, opts), work.Off(ir-1).CMatrix(ldwrkr, opts), cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

						//                    Multiply right singular vectors of L in WORK(IU) by
						//                    Q in VT, storing result in A
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: 0)
						if err = a.Gemm(NoTrans, NoTrans, m, n, m, cone, work.Off(iu-1).CMatrix(ldwrku, opts), vt, czero); err != nil {
							panic(err)
						}

						//                    Copy right singular vectors of A from A to VT
						Zlacpy(Full, m, n, a, vt)

						//                    Copy left singular vectors of A from WORK(IR) to A
						Zlacpy(Full, m, m, work.Off(ir-1).CMatrix(ldwrkr, opts), a)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + m

						//                    Compute A=L*Q, copying result to VT
						//                    (CWorkspace: need 2*M, prefer M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Upper, m, n, a, vt)

						//                    Generate Q in VT
						//                    (CWorkspace: need M+N, prefer M+N*NB)
						//                    (RWorkspace: 0)
						if err = Zunglq(n, n, m, vt, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = 1
						itauq = itau
						itaup = itauq + m
						iwork = itaup + m

						//                    Zero out above L in A
						Zlaset(Upper, m-1, m-1, czero, czero, a.Off(0, 1))

						//                    Bidiagonalize L in A
						//                    (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
						//                    (RWorkspace: need M)
						if err = Zgebrd(m, m, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply right bidiagonalizing vectors in A by Q
						//                    in VT
						//                    (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
						//                    (RWorkspace: 0)
						if err = Zunmbr('P', Left, ConjTrans, m, n, m, a, work.Off(itaup-1), vt, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left bidiagonalizing vectors in A
						//                    (CWorkspace: need 3*M, prefer 2*M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('Q', m, m, m, a, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + m

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in A and computing right
						//                    singular vectors of A in VT
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, m, n, m, 0, s, rwork.Off(ie-1), vt, a, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

					}

				} else if wntuas {
					//                 Path 9t(N much larger than M, JOBU='S' or 'A',
					//                         JOBVT='A')
					//                 N right singular vectors to be computed in VT and
					//                 M left singular vectors to be computed in U
					if lwork >= m*m+max(n+m, 3*m) {
						//                    Sufficient workspace for a fast algorithm
						iu = 1
						if lwork >= wrkbl+a.Rows*m {
							//                       WORK(IU) is LDA by M
							ldwrku = a.Rows
						} else {
							//                       WORK(IU) is M by M
							ldwrku = m
						}
						itau = iu + ldwrku*m
						iwork = itau + m

						//                    Compute A=L*Q, copying result to VT
						//                    (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Upper, m, n, a, vt)

						//                    Generate Q in VT
						//                    (CWorkspace: need M*M+M+N, prefer M*M+M+N*NB)
						//                    (RWorkspace: 0)
						if err = Zunglq(n, n, m, vt, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy L to WORK(IU), zeroing out above it
						Zlacpy(Lower, m, m, a, work.Off(iu-1).CMatrix(ldwrku, opts))
						Zlaset(Upper, m-1, m-1, czero, czero, work.Off(iu+ldwrku-1).CMatrix(ldwrku, opts))
						ie = 1
						itauq = itau
						itaup = itauq + m
						iwork = itaup + m

						//                    Bidiagonalize L in WORK(IU), copying result to U
						//                    (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
						//                    (RWorkspace: need M)
						if err = Zgebrd(m, m, work.Off(iu-1).CMatrix(ldwrku, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Lower, m, m, work.Off(iu-1).CMatrix(ldwrku, opts), u)

						//                    Generate right bidiagonalizing vectors in WORK(IU)
						//                    (CWorkspace: need M*M+3*M, prefer M*M+2*M+(M-1)*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('P', m, m, m, work.Off(iu-1).CMatrix(ldwrku, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left bidiagonalizing vectors in U
						//                    (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('Q', m, m, m, u, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + m

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of L in U and computing right
						//                    singular vectors of L in WORK(IU)
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, m, m, m, 0, s, rwork.Off(ie-1), work.Off(iu-1).CMatrix(ldwrku, opts), u, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

						//                    Multiply right singular vectors of L in WORK(IU) by
						//                    Q in VT, storing result in A
						//                    (CWorkspace: need M*M)
						//                    (RWorkspace: 0)
						if err = a.Gemm(NoTrans, NoTrans, m, n, m, cone, work.Off(iu-1).CMatrix(ldwrku, opts), vt, czero); err != nil {
							panic(err)
						}

						//                    Copy right singular vectors of A from A to VT
						Zlacpy(Full, m, n, a, vt)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + m

						//                    Compute A=L*Q, copying result to VT
						//                    (CWorkspace: need 2*M, prefer M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Zlacpy(Upper, m, n, a, vt)

						//                    Generate Q in VT
						//                    (CWorkspace: need M+N, prefer M+N*NB)
						//                    (RWorkspace: 0)
						if err = Zunglq(n, n, m, vt, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy L to U, zeroing out above it
						Zlacpy(Lower, m, m, a, u)
						Zlaset(Upper, m-1, m-1, czero, czero, u.Off(0, 1))
						ie = 1
						itauq = itau
						itaup = itauq + m
						iwork = itaup + m

						//                    Bidiagonalize L in U
						//                    (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
						//                    (RWorkspace: need M)
						if err = Zgebrd(m, m, u, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply right bidiagonalizing vectors in U by Q
						//                    in VT
						//                    (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
						//                    (RWorkspace: 0)
						if err = Zunmbr('P', Left, ConjTrans, m, n, m, u, work.Off(itaup-1), vt, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left bidiagonalizing vectors in U
						//                    (CWorkspace: need 3*M, prefer 2*M+M*NB)
						//                    (RWorkspace: 0)
						if err = Zungbr('Q', m, m, m, u, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						irwork = ie + m

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in VT
						//                    (CWorkspace: 0)
						//                    (RWorkspace: need BDSPAC)
						if info, err = Zbdsqr(Upper, m, n, m, 0, s, rwork.Off(ie-1), vt, u, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
							panic(err)
						}

					}

				}

			}

		} else {
			//           N .LT. MNTHR
			//
			//           Path 10t(N greater than M, but not much larger)
			//           Reduce to bidiagonal form without LQ decomposition
			ie = 1
			itauq = 1
			itaup = itauq + m
			iwork = itaup + m

			//           Bidiagonalize A
			//           (CWorkspace: need 2*M+N, prefer 2*M+(M+N)*NB)
			//           (RWorkspace: M)
			if err = Zgebrd(m, n, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
				panic(err)
			}
			if wntuas {
				//              If left singular vectors desired in U, copy result to U
				//              and generate left bidiagonalizing vectors in U
				//              (CWorkspace: need 3*M-1, prefer 2*M+(M-1)*NB)
				//              (RWorkspace: 0)
				Zlacpy(Lower, m, m, a, u)
				if err = Zungbr('Q', m, m, n, u, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
			}
			if wntvas {
				//              If right singular vectors desired in VT, copy result to
				//              VT and generate right bidiagonalizing vectors in VT
				//              (CWorkspace: need 2*M+NRVT, prefer 2*M+NRVT*NB)
				//              (RWorkspace: 0)
				Zlacpy(Upper, m, n, a, vt)
				if wntva {
					nrvt = n
				}
				if wntvs {
					nrvt = m
				}
				if err = Zungbr('P', nrvt, n, m, vt, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
			}
			if wntuo {
				//              If left singular vectors desired in A, generate left
				//              bidiagonalizing vectors in A
				//              (CWorkspace: need 3*M-1, prefer 2*M+(M-1)*NB)
				//              (RWorkspace: 0)
				if err = Zungbr('Q', m, m, n, a, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
			}
			if wntvo {
				//              If right singular vectors desired in A, generate right
				//              bidiagonalizing vectors in A
				//              (CWorkspace: need 3*M, prefer 2*M+M*NB)
				//              (RWorkspace: 0)
				if err = Zungbr('P', m, n, m, a, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
			}
			irwork = ie + m
			if wntuas || wntuo {
				nru = m
			}
			if wntun {
				nru = 0
			}
			if wntvas || wntvo {
				ncvt = n
			}
			if wntvn {
				ncvt = 0
			}
			if (!wntuo) && (!wntvo) {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in U and computing right singular
				//              vectors in VT
				//              (CWorkspace: 0)
				//              (RWorkspace: need BDSPAC)
				if info, err = Zbdsqr(Lower, m, ncvt, nru, 0, s, rwork.Off(ie-1), vt, u, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
					panic(err)
				}
			} else if (!wntuo) && wntvo {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in U and computing right singular
				//              vectors in A
				//              (CWorkspace: 0)
				//              (RWorkspace: need BDSPAC)
				if info, err = Zbdsqr(Lower, m, ncvt, nru, 0, s, rwork.Off(ie-1), a, u, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
					panic(err)
				}
			} else {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in A and computing right singular
				//              vectors in VT
				//              (CWorkspace: 0)
				//              (RWorkspace: need BDSPAC)
				if info, err = Zbdsqr(Lower, m, ncvt, nru, 0, s, rwork.Off(ie-1), vt, a, cdum.CMatrix(1, opts), rwork.Off(irwork-1)); err != nil {
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
