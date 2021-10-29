package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgesvd computes the singular value decomposition (SVD) of a real
// M-by-N matrix A, optionally computing the left and/or right singular
// vectors. The SVD is written
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
// Note that the routine returns V**T, not V.
func Dgesvd(jobu, jobvt byte, m, n int, a *mat.Matrix, s *mat.Vector, u, vt *mat.Matrix, work *mat.Vector, lwork int) (info int, err error) {
	var lquery, wntua, wntuas, wntun, wntuo, wntus, wntva, wntvas, wntvn, wntvo, wntvs bool
	var anrm, bignum, eps, one, smlnum, zero float64
	var bdspac, blk, chunk, i, ie, ir, iscl, itau, itaup, itauq, iu, iwork, ldwrkr, ldwrku, lworkDgebrd, lworkDgelqf, lworkDgeqrf, lworkDorgbrP, lworkDorgbrQ, lworkDorglqM, lworkDorglqN, lworkDorgqrM, lworkDorgqrN, maxwrk, minmn, minwrk, mnthr, ncu, ncvt, nru, nrvt, wrkbl int

	dum := vf(1)

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
		err = fmt.Errorf("vt.Rows < 1 || (wntva && vt.Rows < n) || (wntvs && vt.Rows < minmn): jobvt='%c', vt.Rows=%v, m=%v, n=%v", jobvt, vt.Rows, m, n)
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
		if m >= n && minmn > 0 {
			//           Compute space needed for DBDSQR
			mnthr = Ilaenv(6, "Dgesvd", []byte{jobu, jobvt}, m, n, 0, 0)
			bdspac = 5 * n
			//           Compute space needed for DGEQRF
			if err = Dgeqrf(m, n, a, dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDgeqrf = int(dum.Get(0))
			//           Compute space needed for DORGQR
			if err = Dorgqr(m, n, n, a, dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDorgqrN = int(dum.Get(0))
			if err = Dorgqr(m, m, n, a, dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDorgqrM = int(dum.Get(0))
			//           Compute space needed for DGEBRD
			if err = Dgebrd(n, n, a, s, dum, dum, dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDgebrd = int(dum.Get(0))
			//           Compute space needed for DORGBR P
			if err = Dorgbr('P', n, n, n, a, dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDorgbrP = int(dum.Get(0))
			//           Compute space needed for DORGBR Q
			if err = Dorgbr('Q', n, n, n, a, dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDorgbrQ = int(dum.Get(0))
			//
			if m >= mnthr {
				if wntun {
					//                 Path 1 (M much larger than N, JOBU='N')
					maxwrk = n + lworkDgeqrf
					maxwrk = max(maxwrk, 3*n+lworkDgebrd)
					if wntvo || wntvas {
						maxwrk = max(maxwrk, 3*n+lworkDorgbrP)
					}
					maxwrk = max(maxwrk, bdspac)
					minwrk = max(4*n, bdspac)
				} else if wntuo && wntvn {
					//                 Path 2 (M much larger than N, JOBU='O', JOBVT='N')
					wrkbl = n + lworkDgeqrf
					wrkbl = max(wrkbl, n+lworkDorgqrN)
					wrkbl = max(wrkbl, 3*n+lworkDgebrd)
					wrkbl = max(wrkbl, 3*n+lworkDorgbrQ)
					wrkbl = max(wrkbl, bdspac)
					maxwrk = max(n*n+wrkbl, n*n+m*n+n)
					minwrk = max(3*n+m, bdspac)
				} else if wntuo && wntvas {
					//                 Path 3 (M much larger than N, JOBU='O', JOBVT='S' or
					//                 'A')
					wrkbl = n + lworkDgeqrf
					wrkbl = max(wrkbl, n+lworkDorgqrN)
					wrkbl = max(wrkbl, 3*n+lworkDgebrd)
					wrkbl = max(wrkbl, 3*n+lworkDorgbrQ)
					wrkbl = max(wrkbl, 3*n+lworkDorgbrP)
					wrkbl = max(wrkbl, bdspac)
					maxwrk = max(n*n+wrkbl, n*n+m*n+n)
					minwrk = max(3*n+m, bdspac)
				} else if wntus && wntvn {
					//                 Path 4 (M much larger than N, JOBU='S', JOBVT='N')
					wrkbl = n + lworkDgeqrf
					wrkbl = max(wrkbl, n+lworkDorgqrN)
					wrkbl = max(wrkbl, 3*n+lworkDgebrd)
					wrkbl = max(wrkbl, 3*n+lworkDorgbrQ)
					wrkbl = max(wrkbl, bdspac)
					maxwrk = n*n + wrkbl
					minwrk = max(3*n+m, bdspac)
				} else if wntus && wntvo {
					//                 Path 5 (M much larger than N, JOBU='S', JOBVT='O')
					wrkbl = n + lworkDgeqrf
					wrkbl = max(wrkbl, n+lworkDorgqrN)
					wrkbl = max(wrkbl, 3*n+lworkDgebrd)
					wrkbl = max(wrkbl, 3*n+lworkDorgbrQ)
					wrkbl = max(wrkbl, 3*n+lworkDorgbrP)
					wrkbl = max(wrkbl, bdspac)
					maxwrk = 2*n*n + wrkbl
					minwrk = max(3*n+m, bdspac)
				} else if wntus && wntvas {
					//                 Path 6 (M much larger than N, JOBU='S', JOBVT='S' or
					//                 'A')
					wrkbl = n + lworkDgeqrf
					wrkbl = max(wrkbl, n+lworkDorgqrN)
					wrkbl = max(wrkbl, 3*n+lworkDgebrd)
					wrkbl = max(wrkbl, 3*n+lworkDorgbrQ)
					wrkbl = max(wrkbl, 3*n+lworkDorgbrP)
					wrkbl = max(wrkbl, bdspac)
					maxwrk = n*n + wrkbl
					minwrk = max(3*n+m, bdspac)
				} else if wntua && wntvn {
					//                 Path 7 (M much larger than N, JOBU='A', JOBVT='N')
					wrkbl = n + lworkDgeqrf
					wrkbl = max(wrkbl, n+lworkDorgqrM)
					wrkbl = max(wrkbl, 3*n+lworkDgebrd)
					wrkbl = max(wrkbl, 3*n+lworkDorgbrQ)
					wrkbl = max(wrkbl, bdspac)
					maxwrk = n*n + wrkbl
					minwrk = max(3*n+m, bdspac)
				} else if wntua && wntvo {
					//                 Path 8 (M much larger than N, JOBU='A', JOBVT='O')
					wrkbl = n + lworkDgeqrf
					wrkbl = max(wrkbl, n+lworkDorgqrM)
					wrkbl = max(wrkbl, 3*n+lworkDgebrd)
					wrkbl = max(wrkbl, 3*n+lworkDorgbrQ)
					wrkbl = max(wrkbl, 3*n+lworkDorgbrP)
					wrkbl = max(wrkbl, bdspac)
					maxwrk = 2*n*n + wrkbl
					minwrk = max(3*n+m, bdspac)
				} else if wntua && wntvas {
					//                 Path 9 (M much larger than N, JOBU='A', JOBVT='S' or
					//                 'A')
					wrkbl = n + lworkDgeqrf
					wrkbl = max(wrkbl, n+lworkDorgqrM)
					wrkbl = max(wrkbl, 3*n+lworkDgebrd)
					wrkbl = max(wrkbl, 3*n+lworkDorgbrQ)
					wrkbl = max(wrkbl, 3*n+lworkDorgbrP)
					wrkbl = max(wrkbl, bdspac)
					maxwrk = n*n + wrkbl
					minwrk = max(3*n+m, bdspac)
				}
			} else {
				//              Path 10 (M at least N, but not much larger)
				if err = Dgebrd(m, n, a, s, dum, dum, dum, dum, -1); err != nil {
					panic(err)
				}
				lworkDgebrd = int(dum.Get(0))
				maxwrk = 3*n + lworkDgebrd
				if wntus || wntuo {
					if err = Dorgbr('Q', m, n, n, a, dum, dum, -1); err != nil {
						panic(err)
					}
					lworkDorgbrQ = int(dum.Get(0))
					maxwrk = max(maxwrk, 3*n+lworkDorgbrQ)
				}
				if wntua {
					if err = Dorgbr('Q', m, m, n, a, dum, dum, -1); err != nil {
						panic(err)
					}
					lworkDorgbrQ = int(dum.Get(0))
					maxwrk = max(maxwrk, 3*n+lworkDorgbrQ)
				}
				if !wntvn {
					maxwrk = max(maxwrk, 3*n+lworkDorgbrP)
				}
				maxwrk = max(maxwrk, bdspac)
				minwrk = max(3*n+m, bdspac)
			}
		} else if minmn > 0 {
			//           Compute space needed for DBDSQR
			mnthr = Ilaenv(6, "Dgesvd", []byte{jobu, jobvt}, m, n, 0, 0)
			bdspac = 5 * m
			//           Compute space needed for DGELQF
			if err = Dgelqf(m, n, a, dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDgelqf = int(dum.Get(0))
			//           Compute space needed for DORGLQ
			if err = Dorglq(n, n, m, dum.Matrix(n, opts), dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDorglqN = int(dum.Get(0))
			if err = Dorglq(m, n, m, a, dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDorglqM = int(dum.Get(0))
			//           Compute space needed for DGEBRD
			if err = Dgebrd(m, m, a, s, dum, dum, dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDgebrd = int(dum.Get(0))
			//            Compute space needed for DORGBR P
			if err = Dorgbr('P', m, m, m, a, dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDorgbrP = int(dum.Get(0))
			//           Compute space needed for DORGBR Q
			if err = Dorgbr('Q', m, m, m, a, dum, dum, -1); err != nil {
				panic(err)
			}
			lworkDorgbrQ = int(dum.Get(0))
			if n >= mnthr {
				if wntvn {
					//                 Path 1t(N much larger than M, JOBVT='N')
					maxwrk = m + lworkDgelqf
					maxwrk = max(maxwrk, 3*m+lworkDgebrd)
					if wntuo || wntuas {
						maxwrk = max(maxwrk, 3*m+lworkDorgbrQ)
					}
					maxwrk = max(maxwrk, bdspac)
					minwrk = max(4*m, bdspac)
				} else if wntvo && wntun {
					//                 Path 2t(N much larger than M, JOBU='N', JOBVT='O')
					wrkbl = m + lworkDgelqf
					wrkbl = max(wrkbl, m+lworkDorglqM)
					wrkbl = max(wrkbl, 3*m+lworkDgebrd)
					wrkbl = max(wrkbl, 3*m+lworkDorgbrP)
					wrkbl = max(wrkbl, bdspac)
					maxwrk = max(m*m+wrkbl, m*m+m*n+m)
					minwrk = max(3*m+n, bdspac)
				} else if wntvo && wntuas {
					//                 Path 3t(N much larger than M, JOBU='S' or 'A',
					//                 JOBVT='O')
					wrkbl = m + lworkDgelqf
					wrkbl = max(wrkbl, m+lworkDorglqM)
					wrkbl = max(wrkbl, 3*m+lworkDgebrd)
					wrkbl = max(wrkbl, 3*m+lworkDorgbrP)
					wrkbl = max(wrkbl, 3*m+lworkDorgbrQ)
					wrkbl = max(wrkbl, bdspac)
					maxwrk = max(m*m+wrkbl, m*m+m*n+m)
					minwrk = max(3*m+n, bdspac)
				} else if wntvs && wntun {
					//                 Path 4t(N much larger than M, JOBU='N', JOBVT='S')
					wrkbl = m + lworkDgelqf
					wrkbl = max(wrkbl, m+lworkDorglqM)
					wrkbl = max(wrkbl, 3*m+lworkDgebrd)
					wrkbl = max(wrkbl, 3*m+lworkDorgbrP)
					wrkbl = max(wrkbl, bdspac)
					maxwrk = m*m + wrkbl
					minwrk = max(3*m+n, bdspac)
				} else if wntvs && wntuo {
					//                 Path 5t(N much larger than M, JOBU='O', JOBVT='S')
					wrkbl = m + lworkDgelqf
					wrkbl = max(wrkbl, m+lworkDorglqM)
					wrkbl = max(wrkbl, 3*m+lworkDgebrd)
					wrkbl = max(wrkbl, 3*m+lworkDorgbrP)
					wrkbl = max(wrkbl, 3*m+lworkDorgbrQ)
					wrkbl = max(wrkbl, bdspac)
					maxwrk = 2*m*m + wrkbl
					minwrk = max(3*m+n, bdspac)
				} else if wntvs && wntuas {
					//                 Path 6t(N much larger than M, JOBU='S' or 'A',
					//                 JOBVT='S')
					wrkbl = m + lworkDgelqf
					wrkbl = max(wrkbl, m+lworkDorglqM)
					wrkbl = max(wrkbl, 3*m+lworkDgebrd)
					wrkbl = max(wrkbl, 3*m+lworkDorgbrP)
					wrkbl = max(wrkbl, 3*m+lworkDorgbrQ)
					wrkbl = max(wrkbl, bdspac)
					maxwrk = m*m + wrkbl
					minwrk = max(3*m+n, bdspac)
				} else if wntva && wntun {
					//                 Path 7t(N much larger than M, JOBU='N', JOBVT='A')
					wrkbl = m + lworkDgelqf
					wrkbl = max(wrkbl, m+lworkDorglqN)
					wrkbl = max(wrkbl, 3*m+lworkDgebrd)
					wrkbl = max(wrkbl, 3*m+lworkDorgbrP)
					wrkbl = max(wrkbl, bdspac)
					maxwrk = m*m + wrkbl
					minwrk = max(3*m+n, bdspac)
				} else if wntva && wntuo {
					//                 Path 8t(N much larger than M, JOBU='O', JOBVT='A')
					wrkbl = m + lworkDgelqf
					wrkbl = max(wrkbl, m+lworkDorglqN)
					wrkbl = max(wrkbl, 3*m+lworkDgebrd)
					wrkbl = max(wrkbl, 3*m+lworkDorgbrP)
					wrkbl = max(wrkbl, 3*m+lworkDorgbrQ)
					wrkbl = max(wrkbl, bdspac)
					maxwrk = 2*m*m + wrkbl
					minwrk = max(3*m+n, bdspac)
				} else if wntva && wntuas {
					//                 Path 9t(N much larger than M, JOBU='S' or 'A',
					//                 JOBVT='A')
					wrkbl = m + lworkDgelqf
					wrkbl = max(wrkbl, m+lworkDorglqN)
					wrkbl = max(wrkbl, 3*m+lworkDgebrd)
					wrkbl = max(wrkbl, 3*m+lworkDorgbrP)
					wrkbl = max(wrkbl, 3*m+lworkDorgbrQ)
					wrkbl = max(wrkbl, bdspac)
					maxwrk = m*m + wrkbl
					minwrk = max(3*m+n, bdspac)
				}
			} else {
				//              Path 10t(N greater than M, but not much larger)
				if err = Dgebrd(m, n, a, s, dum, dum, dum, dum, -1); err != nil {
					panic(err)
				}
				lworkDgebrd = int(dum.Get(0))
				maxwrk = 3*m + lworkDgebrd
				if wntvs || wntvo {
					//                Compute space needed for DORGBR P
					if err = Dorgbr('P', m, n, m, a, dum, dum, -1); err != nil {
						panic(err)
					}
					lworkDorgbrP = int(dum.Get(0))
					maxwrk = max(maxwrk, 3*m+lworkDorgbrP)
				}
				if wntva {
					if err = Dorgbr('P', n, n, m, a, dum, dum, -1); err != nil {
						panic(err)
					}
					lworkDorgbrP = int(dum.Get(0))
					maxwrk = max(maxwrk, 3*m+lworkDorgbrP)
				}
				if !wntun {
					maxwrk = max(maxwrk, 3*m+lworkDorgbrQ)
				}
				maxwrk = max(maxwrk, bdspac)
				minwrk = max(3*m+n, bdspac)
			}
		}
		maxwrk = max(maxwrk, minwrk)
		work.Set(0, float64(maxwrk))

		if lwork < minwrk && !lquery {
			err = fmt.Errorf("lwork < minwrk && !lquery: lwork=%v, minwrk=%v, lquery=%v", lwork, minwrk, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dgesvd", err)
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

			if wntun {
				//              Path 1 (M much larger than N, JOBU='N')
				//              No left singular vectors to be computed
				itau = 1
				iwork = itau + n

				//              Compute A=Q*R
				//              (Workspace: need 2*N, prefer N + N*NB)
				if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}

				//              Zero out below R
				if n > 1 {
					Dlaset(Lower, n-1, n-1, zero, zero, a.Off(1, 0))
				}
				ie = 1
				itauq = ie + n
				itaup = itauq + n
				iwork = itaup + n

				//              Bidiagonalize R in A
				//              (Workspace: need 4*N, prefer 3*N + 2*N*NB)
				if err = Dgebrd(n, n, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
				ncvt = 0
				if wntvo || wntvas {
					//                 If right singular vectors desired, generate P'.
					//                 (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
					if err = Dorgbr('P', n, n, n, a, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					ncvt = n
				}
				iwork = ie + n

				//              Perform bidiagonal QR iteration, computing right
				//              singular vectors of A in A if desired
				//              (Workspace: need BDSPAC)
				info, err = Dbdsqr(Upper, n, ncvt, 0, 0, s, work.Off(ie-1), a, dum.Matrix(1, opts), dum.Matrix(1, opts), work.Off(iwork-1))

				//              If right singular vectors desired in VT, copy them there
				if wntvas {
					Dlacpy(Full, n, n, a, vt)
				}

			} else if wntuo && wntvn {
				//              Path 2 (M much larger than N, JOBU='O', JOBVT='N')
				//              N left singular vectors to be overwritten on A and
				//              no right singular vectors to be computed
				if lwork >= n*n+max(4*n, bdspac) {
					//                 Sufficient workspace for a fast algorithm
					ir = 1
					if lwork >= max(wrkbl, a.Rows*n+n)+a.Rows*n {
						//                    WORK(IU) is LDA by N, WORK(IR) is LDA by N
						ldwrku = a.Rows
						ldwrkr = a.Rows
					} else if lwork >= max(wrkbl, a.Rows*n+n)+n*n {
						//                    WORK(IU) is LDA by N, WORK(IR) is N by N
						ldwrku = a.Rows
						ldwrkr = n
					} else {
						//                    WORK(IU) is LDWRKU by N, WORK(IR) is N by N
						ldwrku = (lwork - n*n - n) / n
						ldwrkr = n
					}
					itau = ir + ldwrkr*n
					iwork = itau + n

					//                 Compute A=Q*R
					//                 (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
					if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Copy R to WORK(IR) and zero out below it
					Dlacpy(Upper, n, n, a, work.MatrixOff(ir-1, ldwrkr, opts))
					Dlaset(Lower, n-1, n-1, zero, zero, work.MatrixOff(ir, ldwrkr, opts))

					//                 Generate Q in A
					//                 (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
					if err = Dorgqr(m, n, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					ie = itau
					itauq = ie + n
					itaup = itauq + n
					iwork = itaup + n

					//                 Bidiagonalize R in WORK(IR)
					//                 (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
					if err = Dgebrd(n, n, work.MatrixOff(ir-1, ldwrkr, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Generate left vectors bidiagonalizing R
					//                 (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
					if err = Dorgbr('Q', n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					iwork = ie + n

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of R in WORK(IR)
					//                 (Workspace: need N*N + BDSPAC)
					info, err = Dbdsqr(Upper, n, 0, n, 0, s, work.Off(ie-1), dum.Matrix(1, opts), work.MatrixOff(ir-1, ldwrkr, opts), dum.Matrix(1, opts), work.Off(iwork-1))
					iu = ie + n

					//                 Multiply Q in A by left singular vectors of R in
					//                 WORK(IR), storing result in WORK(IU) and copying to A
					//                 (Workspace: need N*N + 2*N, prefer N*N + M*N + N)
					for i = 1; i <= m; i += ldwrku {
						chunk = min(m-i+1, ldwrku)
						err = goblas.Dgemm(NoTrans, NoTrans, chunk, n, n, one, a.Off(i-1, 0), work.MatrixOff(ir-1, ldwrkr, opts), zero, work.MatrixOff(iu-1, ldwrku, opts))
						Dlacpy(Full, chunk, n, work.MatrixOff(iu-1, ldwrku, opts), a.Off(i-1, 0))
					}

				} else {
					//                 Insufficient workspace for a fast algorithm
					ie = 1
					itauq = ie + n
					itaup = itauq + n
					iwork = itaup + n

					//                 Bidiagonalize A
					//                 (Workspace: need 3*N + M, prefer 3*N + (M + N)*NB)
					if err = Dgebrd(m, n, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Generate left vectors bidiagonalizing A
					//                 (Workspace: need 4*N, prefer 3*N + N*NB)
					if err = Dorgbr('Q', m, n, n, a, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					iwork = ie + n

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of A in A
					//                 (Workspace: need BDSPAC)
					info, err = Dbdsqr(Upper, n, 0, m, 0, s, work.Off(ie-1), dum.Matrix(1, opts), a, dum.Matrix(1, opts), work.Off(iwork-1))

				}

			} else if wntuo && wntvas {
				//              Path 3 (M much larger than N, JOBU='O', JOBVT='S' or 'A')
				//              N left singular vectors to be overwritten on A and
				//              N right singular vectors to be computed in VT
				if lwork >= n*n+max(4*n, bdspac) {
					//                 Sufficient workspace for a fast algorithm
					ir = 1
					if lwork >= max(wrkbl, a.Rows*n+n)+a.Rows*n {
						//                    WORK(IU) is LDA by N and WORK(IR) is LDA by N
						ldwrku = a.Rows
						ldwrkr = a.Rows
					} else if lwork >= max(wrkbl, a.Rows*n+n)+n*n {
						//                    WORK(IU) is LDA by N and WORK(IR) is N by N
						ldwrku = a.Rows
						ldwrkr = n
					} else {
						//                    WORK(IU) is LDWRKU by N and WORK(IR) is N by N
						ldwrku = (lwork - n*n - n) / n
						ldwrkr = n
					}
					itau = ir + ldwrkr*n
					iwork = itau + n

					//                 Compute A=Q*R
					//                 (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
					if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Copy R to VT, zeroing out below it
					Dlacpy(Upper, n, n, a, vt)
					if n > 1 {
						Dlaset(Lower, n-1, n-1, zero, zero, vt.Off(1, 0))
					}

					//                 Generate Q in A
					//                 (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
					if err = Dorgqr(m, n, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					ie = itau
					itauq = ie + n
					itaup = itauq + n
					iwork = itaup + n

					//                 Bidiagonalize R in VT, copying result to WORK(IR)
					//                 (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
					if err = Dgebrd(n, n, vt, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					Dlacpy(Lower, n, n, vt, work.MatrixOff(ir-1, ldwrkr, opts))

					//                 Generate left vectors bidiagonalizing R in WORK(IR)
					//                 (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
					if err = Dorgbr('Q', n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Generate right vectors bidiagonalizing R in VT
					//                 (Workspace: need N*N + 4*N-1, prefer N*N + 3*N + (N-1)*NB)
					if err = Dorgbr('P', n, n, n, vt, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					iwork = ie + n

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of R in WORK(IR) and computing right
					//                 singular vectors of R in VT
					//                 (Workspace: need N*N + BDSPAC)
					info, err = Dbdsqr(Upper, n, n, n, 0, s, work.Off(ie-1), vt, work.MatrixOff(ir-1, ldwrkr, opts), dum.Matrix(1, opts), work.Off(iwork-1))
					iu = ie + n

					//                 Multiply Q in A by left singular vectors of R in
					//                 WORK(IR), storing result in WORK(IU) and copying to A
					//                 (Workspace: need N*N + 2*N, prefer N*N + M*N + N)
					for i = 1; i <= m; i += ldwrku {
						chunk = min(m-i+1, ldwrku)
						err = goblas.Dgemm(NoTrans, NoTrans, chunk, n, n, one, a.Off(i-1, 0), work.MatrixOff(ir-1, ldwrkr, opts), zero, work.MatrixOff(iu-1, ldwrku, opts))
						Dlacpy(Full, chunk, n, work.MatrixOff(iu-1, ldwrku, opts), a.Off(i-1, 0))
					}

				} else {
					//                 Insufficient workspace for a fast algorithm
					itau = 1
					iwork = itau + n

					//                 Compute A=Q*R
					//                 (Workspace: need 2*N, prefer N + N*NB)
					if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Copy R to VT, zeroing out below it
					Dlacpy(Upper, n, n, a, vt)
					if n > 1 {
						Dlaset(Lower, n-1, n-1, zero, zero, vt.Off(1, 0))
					}

					//                 Generate Q in A
					//                 (Workspace: need 2*N, prefer N + N*NB)
					if err = Dorgqr(m, n, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					ie = itau
					itauq = ie + n
					itaup = itauq + n
					iwork = itaup + n

					//                 Bidiagonalize R in VT
					//                 (Workspace: need 4*N, prefer 3*N + 2*N*NB)
					if err = Dgebrd(n, n, vt, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Multiply Q in A by left vectors bidiagonalizing R
					//                 (Workspace: need 3*N + M, prefer 3*N + M*NB)
					if err = Dormbr('Q', Right, NoTrans, m, n, n, vt, work.Off(itauq-1), a, work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Generate right vectors bidiagonalizing R in VT
					//                 (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
					if err = Dorgbr('P', n, n, n, vt, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					iwork = ie + n

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of A in A and computing right
					//                 singular vectors of A in VT
					//                 (Workspace: need BDSPAC)
					info, err = Dbdsqr(Upper, n, n, m, 0, s, work.Off(ie-1), vt, a, dum.Matrix(1, opts), work.Off(iwork-1))

				}

			} else if wntus {

				if wntvn {
					//                 Path 4 (M much larger than N, JOBU='S', JOBVT='N')
					//                 N left singular vectors to be computed in U and
					//                 no right singular vectors to be computed
					if lwork >= n*n+max(4*n, bdspac) {
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
						//                    (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
						if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy R to WORK(IR), zeroing out below it
						Dlacpy(Upper, n, n, a, work.MatrixOff(ir-1, ldwrkr, opts))
						Dlaset(Lower, n-1, n-1, zero, zero, work.MatrixOff(ir, ldwrkr, opts))

						//                    Generate Q in A
						//                    (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
						if err = Dorgqr(m, n, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = itau
						itauq = ie + n
						itaup = itauq + n
						iwork = itaup + n

						//                    Bidiagonalize R in WORK(IR)
						//                    (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
						if err = Dgebrd(n, n, work.MatrixOff(ir-1, ldwrkr, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left vectors bidiagonalizing R in WORK(IR)
						//                    (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
						if err = Dorgbr('Q', n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IR)
						//                    (Workspace: need N*N + BDSPAC)
						info, err = Dbdsqr(Upper, n, 0, n, 0, s, work.Off(ie-1), dum.Matrix(1, opts), work.MatrixOff(ir-1, ldwrkr, opts), dum.Matrix(1, opts), work.Off(iwork-1))

						//                    Multiply Q in A by left singular vectors of R in
						//                    WORK(IR), storing result in U
						//                    (Workspace: need N*N)
						err = goblas.Dgemm(NoTrans, NoTrans, m, n, n, one, a, work.MatrixOff(ir-1, ldwrkr, opts), zero, u)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + n

						//                    Compute A=Q*R, copying result to U
						//                    (Workspace: need 2*N, prefer N + N*NB)
						if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Lower, m, n, a, u)

						//                    Generate Q in U
						//                    (Workspace: need 2*N, prefer N + N*NB)
						if err = Dorgqr(m, n, n, u, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = itau
						itauq = ie + n
						itaup = itauq + n
						iwork = itaup + n

						//                    Zero out below R in A
						if n > 1 {
							Dlaset(Lower, n-1, n-1, zero, zero, a.Off(1, 0))
						}

						//                    Bidiagonalize R in A
						//                    (Workspace: need 4*N, prefer 3*N + 2*N*NB)
						if err = Dgebrd(n, n, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply Q in U by left vectors bidiagonalizing R
						//                    (Workspace: need 3*N + M, prefer 3*N + M*NB)
						if err = Dormbr('Q', Right, NoTrans, m, n, n, a, work.Off(itauq-1), u, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U
						//                    (Workspace: need BDSPAC)
						info, err = Dbdsqr(Upper, n, 0, m, 0, s, work.Off(ie-1), dum.Matrix(1, opts), u, dum.Matrix(1, opts), work.Off(iwork-1))

					}

				} else if wntvo {
					//                 Path 5 (M much larger than N, JOBU='S', JOBVT='O')
					//                 N left singular vectors to be computed in U and
					//                 N right singular vectors to be overwritten on A
					if lwork >= 2*n*n+max(4*n, bdspac) {
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
						//                    (Workspace: need 2*N*N + 2*N, prefer 2*N*N + N + N*NB)
						if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy R to WORK(IU), zeroing out below it
						Dlacpy(Upper, n, n, a, work.MatrixOff(iu-1, ldwrku, opts))
						Dlaset(Lower, n-1, n-1, zero, zero, work.MatrixOff(iu, ldwrku, opts))

						//                    Generate Q in A
						//                    (Workspace: need 2*N*N + 2*N, prefer 2*N*N + N + N*NB)
						if err = Dorgqr(m, n, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = itau
						itauq = ie + n
						itaup = itauq + n
						iwork = itaup + n

						//                    Bidiagonalize R in WORK(IU), copying result to
						//                    WORK(IR)
						//                    (Workspace: need 2*N*N + 4*N,
						//                                prefer 2*N*N+3*N+2*N*NB)
						if err = Dgebrd(n, n, work.MatrixOff(iu-1, ldwrku, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Upper, n, n, work.MatrixOff(iu-1, ldwrku, opts), work.MatrixOff(ir-1, ldwrkr, opts))

						//                    Generate left bidiagonalizing vectors in WORK(IU)
						//                    (Workspace: need 2*N*N + 4*N, prefer 2*N*N + 3*N + N*NB)
						if err = Dorgbr('Q', n, n, n, work.MatrixOff(iu-1, ldwrku, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right bidiagonalizing vectors in WORK(IR)
						//                    (Workspace: need 2*N*N + 4*N-1,
						//                                prefer 2*N*N+3*N+(N-1)*NB)
						if err = Dorgbr('P', n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IU) and computing
						//                    right singular vectors of R in WORK(IR)
						//                    (Workspace: need 2*N*N + BDSPAC)
						info, err = Dbdsqr(Upper, n, n, n, 0, s, work.Off(ie-1), work.MatrixOff(ir-1, ldwrkr, opts), work.MatrixOff(iu-1, ldwrku, opts), dum.Matrix(1, opts), work.Off(iwork-1))

						//                    Multiply Q in A by left singular vectors of R in
						//                    WORK(IU), storing result in U
						//                    (Workspace: need N*N)
						err = goblas.Dgemm(NoTrans, NoTrans, m, n, n, one, a, work.MatrixOff(iu-1, ldwrku, opts), zero, u)

						//                    Copy right singular vectors of R to A
						//                    (Workspace: need N*N)
						Dlacpy(Full, n, n, work.MatrixOff(ir-1, ldwrkr, opts), a)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + n

						//                    Compute A=Q*R, copying result to U
						//                    (Workspace: need 2*N, prefer N + N*NB)
						if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Lower, m, n, a, u)

						//                    Generate Q in U
						//                    (Workspace: need 2*N, prefer N + N*NB)
						if err = Dorgqr(m, n, n, u, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = itau
						itauq = ie + n
						itaup = itauq + n
						iwork = itaup + n

						//                    Zero out below R in A
						if n > 1 {
							Dlaset(Lower, n-1, n-1, zero, zero, a.Off(1, 0))
						}

						//                    Bidiagonalize R in A
						//                    (Workspace: need 4*N, prefer 3*N + 2*N*NB)
						if err = Dgebrd(n, n, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply Q in U by left vectors bidiagonalizing R
						//                    (Workspace: need 3*N + M, prefer 3*N + M*NB)
						if err = Dormbr('Q', Right, NoTrans, m, n, n, a, work.Off(itauq-1), u, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right vectors bidiagonalizing R in A
						//                    (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
						if err = Dorgbr('P', n, n, n, a, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in A
						//                    (Workspace: need BDSPAC)
						info, err = Dbdsqr(Upper, n, n, m, 0, s, work.Off(ie-1), a, u, dum.Matrix(1, opts), work.Off(iwork-1))

					}

				} else if wntvas {
					//                 Path 6 (M much larger than N, JOBU='S', JOBVT='S'
					//                         or 'A')
					//                 N left singular vectors to be computed in U and
					//                 N right singular vectors to be computed in VT
					if lwork >= n*n+max(4*n, bdspac) {
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
						//                    (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
						if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy R to WORK(IU), zeroing out below it
						Dlacpy(Upper, n, n, a, work.MatrixOff(iu-1, ldwrku, opts))
						Dlaset(Lower, n-1, n-1, zero, zero, work.MatrixOff(iu, ldwrku, opts))

						//                    Generate Q in A
						//                    (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
						if err = Dorgqr(m, n, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = itau
						itauq = ie + n
						itaup = itauq + n
						iwork = itaup + n

						//                    Bidiagonalize R in WORK(IU), copying result to VT
						//                    (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
						if err = Dgebrd(n, n, work.MatrixOff(iu-1, ldwrku, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Upper, n, n, work.MatrixOff(iu-1, ldwrku, opts), vt)

						//                    Generate left bidiagonalizing vectors in WORK(IU)
						//                    (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
						if err = Dorgbr('Q', n, n, n, work.MatrixOff(iu-1, ldwrku, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right bidiagonalizing vectors in VT
						//                    (Workspace: need N*N + 4*N-1,
						//                                prefer N*N+3*N+(N-1)*NB)
						if err = Dorgbr('P', n, n, n, vt, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IU) and computing
						//                    right singular vectors of R in VT
						//                    (Workspace: need N*N + BDSPAC)
						info, err = Dbdsqr(Upper, n, n, n, 0, s, work.Off(ie-1), vt, work.MatrixOff(iu-1, ldwrku, opts), dum.Matrix(1, opts), work.Off(iwork-1))

						//                    Multiply Q in A by left singular vectors of R in
						//                    WORK(IU), storing result in U
						//                    (Workspace: need N*N)
						err = goblas.Dgemm(NoTrans, NoTrans, m, n, n, one, a, work.MatrixOff(iu-1, ldwrku, opts), zero, u)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + n

						//                    Compute A=Q*R, copying result to U
						//                    (Workspace: need 2*N, prefer N + N*NB)
						if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Lower, m, n, a, u)

						//                    Generate Q in U
						//                    (Workspace: need 2*N, prefer N + N*NB)
						if err = Dorgqr(m, n, n, u, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy R to VT, zeroing out below it
						Dlacpy(Upper, n, n, a, vt)
						if n > 1 {
							Dlaset(Lower, n-1, n-1, zero, zero, vt.Off(1, 0))
						}
						ie = itau
						itauq = ie + n
						itaup = itauq + n
						iwork = itaup + n

						//                    Bidiagonalize R in VT
						//                    (Workspace: need 4*N, prefer 3*N + 2*N*NB)
						if err = Dgebrd(n, n, vt, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply Q in U by left bidiagonalizing vectors
						//                    in VT
						//                    (Workspace: need 3*N + M, prefer 3*N + M*NB)
						if err = Dormbr('Q', Right, NoTrans, m, n, n, vt, work.Off(itauq-1), u, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right bidiagonalizing vectors in VT
						//                    (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
						if err = Dorgbr('P', n, n, n, vt, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in VT
						//                    (Workspace: need BDSPAC)
						info, err = Dbdsqr(Upper, n, n, m, 0, s, work.Off(ie-1), vt, u, dum.Matrix(1, opts), work.Off(iwork-1))

					}

				}

			} else if wntua {

				if wntvn {
					//                 Path 7 (M much larger than N, JOBU='A', JOBVT='N')
					//                 M left singular vectors to be computed in U and
					//                 no right singular vectors to be computed
					if lwork >= n*n+max(n+m, 4*n, bdspac) {
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
						//                    (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
						if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Lower, m, n, a, u)

						//                    Copy R to WORK(IR), zeroing out below it
						Dlacpy(Upper, n, n, a, work.MatrixOff(ir-1, ldwrkr, opts))
						Dlaset(Lower, n-1, n-1, zero, zero, work.MatrixOff(ir, ldwrkr, opts))

						//                    Generate Q in U
						//                    (Workspace: need N*N + N + M, prefer N*N + N + M*NB)
						if err = Dorgqr(m, m, n, u, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = itau
						itauq = ie + n
						itaup = itauq + n
						iwork = itaup + n

						//                    Bidiagonalize R in WORK(IR)
						//                    (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
						if err = Dgebrd(n, n, work.MatrixOff(ir-1, ldwrkr, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left bidiagonalizing vectors in WORK(IR)
						//                    (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
						if err = Dorgbr('Q', n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IR)
						//                    (Workspace: need N*N + BDSPAC)
						info, err = Dbdsqr(Upper, n, 0, n, 0, s, work.Off(ie-1), dum.Matrix(1, opts), work.MatrixOff(ir-1, ldwrkr, opts), dum.Matrix(1, opts), work.Off(iwork-1))

						//                    Multiply Q in U by left singular vectors of R in
						//                    WORK(IR), storing result in A
						//                    (Workspace: need N*N)
						err = goblas.Dgemm(NoTrans, NoTrans, m, n, n, one, u, work.MatrixOff(ir-1, ldwrkr, opts), zero, a)

						//                    Copy left singular vectors of A from A to U
						Dlacpy(Full, m, n, a, u)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + n

						//                    Compute A=Q*R, copying result to U
						//                    (Workspace: need 2*N, prefer N + N*NB)
						if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Lower, m, n, a, u)

						//                    Generate Q in U
						//                    (Workspace: need N + M, prefer N + M*NB)
						if err = Dorgqr(m, m, n, u, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = itau
						itauq = ie + n
						itaup = itauq + n
						iwork = itaup + n

						//                    Zero out below R in A
						if n > 1 {
							Dlaset(Lower, n-1, n-1, zero, zero, a.Off(1, 0))
						}

						//                    Bidiagonalize R in A
						//                    (Workspace: need 4*N, prefer 3*N + 2*N*NB)
						if err = Dgebrd(n, n, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply Q in U by left bidiagonalizing vectors
						//                    in A
						//                    (Workspace: need 3*N + M, prefer 3*N + M*NB)
						if err = Dormbr('Q', Right, NoTrans, m, n, n, a, work.Off(itauq-1), u, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U
						//                    (Workspace: need BDSPAC)
						info, err = Dbdsqr(Upper, n, 0, m, 0, s, work.Off(ie-1), dum.Matrix(1, opts), u, dum.Matrix(1, opts), work.Off(iwork-1))

					}

				} else if wntvo {
					//                 Path 8 (M much larger than N, JOBU='A', JOBVT='O')
					//                 M left singular vectors to be computed in U and
					//                 N right singular vectors to be overwritten on A
					if lwork >= 2*n*n+max(n+m, 4*n, bdspac) {
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
						//                    (Workspace: need 2*N*N + 2*N, prefer 2*N*N + N + N*NB)
						if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Lower, m, n, a, u)

						//                    Generate Q in U
						//                    (Workspace: need 2*N*N + N + M, prefer 2*N*N + N + M*NB)
						if err = Dorgqr(m, m, n, u, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy R to WORK(IU), zeroing out below it
						Dlacpy(Upper, n, n, a, work.MatrixOff(iu-1, ldwrku, opts))
						Dlaset(Lower, n-1, n-1, zero, zero, work.MatrixOff(iu, ldwrku, opts))
						ie = itau
						itauq = ie + n
						itaup = itauq + n
						iwork = itaup + n

						//                    Bidiagonalize R in WORK(IU), copying result to
						//                    WORK(IR)
						//                    (Workspace: need 2*N*N + 4*N,
						//                                prefer 2*N*N+3*N+2*N*NB)
						if err = Dgebrd(n, n, work.MatrixOff(iu-1, ldwrku, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Upper, n, n, work.MatrixOff(iu-1, ldwrku, opts), work.MatrixOff(ir-1, ldwrkr, opts))

						//                    Generate left bidiagonalizing vectors in WORK(IU)
						//                    (Workspace: need 2*N*N + 4*N, prefer 2*N*N + 3*N + N*NB)
						if err = Dorgbr('Q', n, n, n, work.MatrixOff(iu-1, ldwrku, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right bidiagonalizing vectors in WORK(IR)
						//                    (Workspace: need 2*N*N + 4*N-1,
						//                                prefer 2*N*N+3*N+(N-1)*NB)
						if err = Dorgbr('P', n, n, n, work.MatrixOff(ir-1, ldwrkr, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IU) and computing
						//                    right singular vectors of R in WORK(IR)
						//                    (Workspace: need 2*N*N + BDSPAC)
						info, err = Dbdsqr(Upper, n, n, n, 0, s, work.Off(ie-1), work.MatrixOff(ir-1, ldwrkr, opts), work.MatrixOff(iu-1, ldwrku, opts), dum.Matrix(1, opts), work.Off(iwork-1))

						//                    Multiply Q in U by left singular vectors of R in
						//                    WORK(IU), storing result in A
						//                    (Workspace: need N*N)
						err = goblas.Dgemm(NoTrans, NoTrans, m, n, n, one, u, work.MatrixOff(iu-1, ldwrku, opts), zero, a)

						//                    Copy left singular vectors of A from A to U
						Dlacpy(Full, m, n, a, u)

						//                    Copy right singular vectors of R from WORK(IR) to A
						Dlacpy(Full, n, n, work.MatrixOff(ir-1, ldwrkr, opts), a)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + n

						//                    Compute A=Q*R, copying result to U
						//                    (Workspace: need 2*N, prefer N + N*NB)
						if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Lower, m, n, a, u)

						//                    Generate Q in U
						//                    (Workspace: need N + M, prefer N + M*NB)
						if err = Dorgqr(m, m, n, u, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = itau
						itauq = ie + n
						itaup = itauq + n
						iwork = itaup + n

						//                    Zero out below R in A
						if n > 1 {
							Dlaset(Lower, n-1, n-1, zero, zero, a.Off(1, 0))
						}

						//                    Bidiagonalize R in A
						//                    (Workspace: need 4*N, prefer 3*N + 2*N*NB)
						if err = Dgebrd(n, n, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply Q in U by left bidiagonalizing vectors
						//                    in A
						//                    (Workspace: need 3*N + M, prefer 3*N + M*NB)
						if err = Dormbr('Q', Right, NoTrans, m, n, n, a, work.Off(itauq-1), u, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right bidiagonalizing vectors in A
						//                    (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
						if err = Dorgbr('P', n, n, n, a, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in A
						//                    (Workspace: need BDSPAC)
						info, err = Dbdsqr(Upper, n, n, m, 0, s, work.Off(ie-1), a, u, dum.Matrix(1, opts), work.Off(iwork-1))

					}

				} else if wntvas {
					//                 Path 9 (M much larger than N, JOBU='A', JOBVT='S'
					//                         or 'A')
					//                 M left singular vectors to be computed in U and
					//                 N right singular vectors to be computed in VT
					if lwork >= n*n+max(n+m, 4*n, bdspac) {
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
						//                    (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
						if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Lower, m, n, a, u)

						//                    Generate Q in U
						//                    (Workspace: need N*N + N + M, prefer N*N + N + M*NB)
						if err = Dorgqr(m, m, n, u, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy R to WORK(IU), zeroing out below it
						Dlacpy(Upper, n, n, a, work.MatrixOff(iu-1, ldwrku, opts))
						Dlaset(Lower, n-1, n-1, zero, zero, work.MatrixOff(iu, ldwrku, opts))
						ie = itau
						itauq = ie + n
						itaup = itauq + n
						iwork = itaup + n

						//                    Bidiagonalize R in WORK(IU), copying result to VT
						//                    (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
						if err = Dgebrd(n, n, work.MatrixOff(iu-1, ldwrku, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Upper, n, n, work.MatrixOff(iu-1, ldwrku, opts), vt)

						//                    Generate left bidiagonalizing vectors in WORK(IU)
						//                    (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
						if err = Dorgbr('Q', n, n, n, work.MatrixOff(iu-1, ldwrku, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right bidiagonalizing vectors in VT
						//                    (Workspace: need N*N + 4*N-1,
						//                                prefer N*N+3*N+(N-1)*NB)
						if err = Dorgbr('P', n, n, n, vt, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of R in WORK(IU) and computing
						//                    right singular vectors of R in VT
						//                    (Workspace: need N*N + BDSPAC)
						info, err = Dbdsqr(Upper, n, n, n, 0, s, work.Off(ie-1), vt, work.MatrixOff(iu-1, ldwrku, opts), dum.Matrix(1, opts), work.Off(iwork-1))

						//                    Multiply Q in U by left singular vectors of R in
						//                    WORK(IU), storing result in A
						//                    (Workspace: need N*N)
						err = goblas.Dgemm(NoTrans, NoTrans, m, n, n, one, u, work.MatrixOff(iu-1, ldwrku, opts), zero, a)

						//                    Copy left singular vectors of A from A to U
						Dlacpy(Full, m, n, a, u)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + n

						//                    Compute A=Q*R, copying result to U
						//                    (Workspace: need 2*N, prefer N + N*NB)
						if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Lower, m, n, a, u)

						//                    Generate Q in U
						//                    (Workspace: need N + M, prefer N + M*NB)
						if err = Dorgqr(m, m, n, u, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy R from A to VT, zeroing out below it
						Dlacpy(Upper, n, n, a, vt)
						if n > 1 {
							Dlaset(Lower, n-1, n-1, zero, zero, vt.Off(1, 0))
						}
						ie = itau
						itauq = ie + n
						itaup = itauq + n
						iwork = itaup + n

						//                    Bidiagonalize R in VT
						//                    (Workspace: need 4*N, prefer 3*N + 2*N*NB)
						if err = Dgebrd(n, n, vt, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply Q in U by left bidiagonalizing vectors
						//                    in VT
						//                    (Workspace: need 3*N + M, prefer 3*N + M*NB)
						if err = Dormbr('Q', Right, NoTrans, m, n, n, vt, work.Off(itauq-1), u, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right bidiagonalizing vectors in VT
						//                    (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
						if err = Dorgbr('P', n, n, n, vt, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + n

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in VT
						//                    (Workspace: need BDSPAC)
						info, err = Dbdsqr(Upper, n, n, m, 0, s, work.Off(ie-1), vt, u, dum.Matrix(1, opts), work.Off(iwork-1))

					}

				}

			}

		} else {
			//           M .LT. MNTHR
			//
			//           Path 10 (M at least N, but not much larger)
			//           Reduce to bidiagonal form without QR decomposition
			ie = 1
			itauq = ie + n
			itaup = itauq + n
			iwork = itaup + n

			//           Bidiagonalize A
			//           (Workspace: need 3*N + M, prefer 3*N + (M + N)*NB)
			if err = Dgebrd(m, n, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
				panic(err)
			}
			if wntuas {
				//              If left singular vectors desired in U, copy result to U
				//              and generate left bidiagonalizing vectors in U
				//              (Workspace: need 3*N + NCU, prefer 3*N + NCU*NB)
				Dlacpy(Lower, m, n, a, u)
				if wntus {
					ncu = n
				}
				if wntua {
					ncu = m
				}
				if err = Dorgbr('Q', m, ncu, n, u, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
			}
			if wntvas {
				//              If right singular vectors desired in VT, copy result to
				//              VT and generate right bidiagonalizing vectors in VT
				//              (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
				Dlacpy(Upper, n, n, a, vt)
				if err = Dorgbr('P', n, n, n, vt, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
			}
			if wntuo {
				//              If left singular vectors desired in A, generate left
				//              bidiagonalizing vectors in A
				//              (Workspace: need 4*N, prefer 3*N + N*NB)
				if err = Dorgbr('Q', m, n, n, a, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
			}
			if wntvo {
				//              If right singular vectors desired in A, generate right
				//              bidiagonalizing vectors in A
				//              (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
				if err = Dorgbr('P', n, n, n, a, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
			}
			iwork = ie + n
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
				//              (Workspace: need BDSPAC)
				info, err = Dbdsqr(Upper, n, ncvt, nru, 0, s, work.Off(ie-1), vt, u, dum.Matrix(1, opts), work.Off(iwork-1))
			} else if (!wntuo) && wntvo {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in U and computing right singular
				//              vectors in A
				//              (Workspace: need BDSPAC)
				info, err = Dbdsqr(Upper, n, ncvt, nru, 0, s, work.Off(ie-1), a, u, dum.Matrix(1, opts), work.Off(iwork-1))
			} else {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in A and computing right singular
				//              vectors in VT
				//              (Workspace: need BDSPAC)
				info, err = Dbdsqr(Upper, n, ncvt, nru, 0, s, work.Off(ie-1), vt, a, dum.Matrix(1, opts), work.Off(iwork-1))
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
				//              (Workspace: need 2*M, prefer M + M*NB)
				if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}

				//              Zero out above L
				Dlaset(Upper, m-1, m-1, zero, zero, a.Off(0, 1))
				ie = 1
				itauq = ie + m
				itaup = itauq + m
				iwork = itaup + m

				//              Bidiagonalize L in A
				//              (Workspace: need 4*M, prefer 3*M + 2*M*NB)
				if err = Dgebrd(m, m, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
				if wntuo || wntuas {
					//                 If left singular vectors desired, generate Q
					//                 (Workspace: need 4*M, prefer 3*M + M*NB)
					if err = Dorgbr('Q', m, m, m, a, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
				}
				iwork = ie + m
				nru = 0
				if wntuo || wntuas {
					nru = m
				}

				//              Perform bidiagonal QR iteration, computing left singular
				//              vectors of A in A if desired
				//              (Workspace: need BDSPAC)
				info, err = Dbdsqr(Upper, m, 0, nru, 0, s, work.Off(ie-1), dum.Matrix(1, opts), a, dum.Matrix(1, opts), work.Off(iwork-1))

				//              If left singular vectors desired in U, copy them there
				if wntuas {
					Dlacpy(Full, m, m, a, u)
				}

			} else if wntvo && wntun {
				//              Path 2t(N much larger than M, JOBU='N', JOBVT='O')
				//              M right singular vectors to be overwritten on A and
				//              no left singular vectors to be computed
				if lwork >= m*m+max(4*m, bdspac) {
					//                 Sufficient workspace for a fast algorithm
					ir = 1
					if lwork >= max(wrkbl, a.Rows*n+m)+a.Rows*m {
						//                    WORK(IU) is LDA by N and WORK(IR) is LDA by M
						ldwrku = a.Rows
						chunk = n
						ldwrkr = a.Rows
					} else if lwork >= max(wrkbl, a.Rows*n+m)+m*m {
						//                    WORK(IU) is LDA by N and WORK(IR) is M by M
						ldwrku = a.Rows
						chunk = n
						ldwrkr = m
					} else {
						//                    WORK(IU) is M by CHUNK and WORK(IR) is M by M
						ldwrku = m
						chunk = (lwork - m*m - m) / m
						ldwrkr = m
					}
					itau = ir + ldwrkr*m
					iwork = itau + m

					//                 Compute A=L*Q
					//                 (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
					if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Copy L to WORK(IR) and zero out above it
					Dlacpy(Lower, m, m, a, work.MatrixOff(ir-1, ldwrkr, opts))
					Dlaset(Upper, m-1, m-1, zero, zero, work.MatrixOff(ir+ldwrkr-1, ldwrkr, opts))

					//                 Generate Q in A
					//                 (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
					if err = Dorglq(m, n, m, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					ie = itau
					itauq = ie + m
					itaup = itauq + m
					iwork = itaup + m

					//                 Bidiagonalize L in WORK(IR)
					//                 (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
					if err = Dgebrd(m, m, work.MatrixOff(ir-1, ldwrkr, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Generate right vectors bidiagonalizing L
					//                 (Workspace: need M*M + 4*M-1, prefer M*M + 3*M + (M-1)*NB)
					if err = Dorgbr('P', m, m, m, work.MatrixOff(ir-1, ldwrkr, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					iwork = ie + m

					//                 Perform bidiagonal QR iteration, computing right
					//                 singular vectors of L in WORK(IR)
					//                 (Workspace: need M*M + BDSPAC)
					info, err = Dbdsqr(Upper, m, m, 0, 0, s, work.Off(ie-1), work.MatrixOff(ir-1, ldwrkr, opts), dum.Matrix(1, opts), dum.Matrix(1, opts), work.Off(iwork-1))
					iu = ie + m

					//                 Multiply right singular vectors of L in WORK(IR) by Q
					//                 in A, storing result in WORK(IU) and copying to A
					//                 (Workspace: need M*M + 2*M, prefer M*M + M*N + M)
					for i = 1; i <= n; i += chunk {
						blk = min(n-i+1, chunk)
						err = goblas.Dgemm(NoTrans, NoTrans, m, blk, m, one, work.MatrixOff(ir-1, ldwrkr, opts), a.Off(0, i-1), zero, work.MatrixOff(iu-1, ldwrku, opts))
						Dlacpy(Full, m, blk, work.MatrixOff(iu-1, ldwrku, opts), a.Off(0, i-1))
					}

				} else {
					//                 Insufficient workspace for a fast algorithm
					ie = 1
					itauq = ie + m
					itaup = itauq + m
					iwork = itaup + m

					//                 Bidiagonalize A
					//                 (Workspace: need 3*M + N, prefer 3*M + (M + N)*NB)
					if err = Dgebrd(m, n, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Generate right vectors bidiagonalizing A
					//                 (Workspace: need 4*M, prefer 3*M + M*NB)
					if err = Dorgbr('P', m, n, m, a, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					iwork = ie + m

					//                 Perform bidiagonal QR iteration, computing right
					//                 singular vectors of A in A
					//                 (Workspace: need BDSPAC)
					info, err = Dbdsqr(Lower, m, n, 0, 0, s, work.Off(ie-1), a, dum.Matrix(1, opts), dum.Matrix(1, opts), work.Off(iwork-1))

				}

			} else if wntvo && wntuas {
				//              Path 3t(N much larger than M, JOBU='S' or 'A', JOBVT='O')
				//              M right singular vectors to be overwritten on A and
				//              M left singular vectors to be computed in U
				if lwork >= m*m+max(4*m, bdspac) {
					//                 Sufficient workspace for a fast algorithm
					ir = 1
					if lwork >= max(wrkbl, a.Rows*n+m)+a.Rows*m {
						//                    WORK(IU) is LDA by N and WORK(IR) is LDA by M
						ldwrku = a.Rows
						chunk = n
						ldwrkr = a.Rows
					} else if lwork >= max(wrkbl, a.Rows*n+m)+m*m {
						//                    WORK(IU) is LDA by N and WORK(IR) is M by M
						ldwrku = a.Rows
						chunk = n
						ldwrkr = m
					} else {
						//                    WORK(IU) is M by CHUNK and WORK(IR) is M by M
						ldwrku = m
						chunk = (lwork - m*m - m) / m
						ldwrkr = m
					}
					itau = ir + ldwrkr*m
					iwork = itau + m

					//                 Compute A=L*Q
					//                 (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
					if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Copy L to U, zeroing about above it
					Dlacpy(Lower, m, m, a, u)
					Dlaset(Upper, m-1, m-1, zero, zero, u.Off(0, 1))

					//                 Generate Q in A
					//                 (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
					if err = Dorglq(m, n, m, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					ie = itau
					itauq = ie + m
					itaup = itauq + m
					iwork = itaup + m

					//                 Bidiagonalize L in U, copying result to WORK(IR)
					//                 (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
					if err = Dgebrd(m, m, u, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					Dlacpy(Upper, m, m, u, work.MatrixOff(ir-1, ldwrkr, opts))

					//                 Generate right vectors bidiagonalizing L in WORK(IR)
					//                 (Workspace: need M*M + 4*M-1, prefer M*M + 3*M + (M-1)*NB)
					if err = Dorgbr('P', m, m, m, work.MatrixOff(ir-1, ldwrkr, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Generate left vectors bidiagonalizing L in U
					//                 (Workspace: need M*M + 4*M, prefer M*M + 3*M + M*NB)
					if err = Dorgbr('Q', m, m, m, u, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					iwork = ie + m

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of L in U, and computing right
					//                 singular vectors of L in WORK(IR)
					//                 (Workspace: need M*M + BDSPAC)
					info, err = Dbdsqr(Upper, m, m, m, 0, s, work.Off(ie-1), work.MatrixOff(ir-1, ldwrkr, opts), u, dum.Matrix(1, opts), work.Off(iwork-1))
					iu = ie + m

					//                 Multiply right singular vectors of L in WORK(IR) by Q
					//                 in A, storing result in WORK(IU) and copying to A
					//                 (Workspace: need M*M + 2*M, prefer M*M + M*N + M))
					for i = 1; i <= n; i += chunk {
						blk = min(n-i+1, chunk)
						err = goblas.Dgemm(NoTrans, NoTrans, m, blk, m, one, work.MatrixOff(ir-1, ldwrkr, opts), a.Off(0, i-1), zero, work.MatrixOff(iu-1, ldwrku, opts))
						Dlacpy(Full, m, blk, work.MatrixOff(iu-1, ldwrku, opts), a.Off(0, i-1))
					}

				} else {
					//                 Insufficient workspace for a fast algorithm
					itau = 1
					iwork = itau + m

					//                 Compute A=L*Q
					//                 (Workspace: need 2*M, prefer M + M*NB)
					if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Copy L to U, zeroing out above it
					Dlacpy(Lower, m, m, a, u)
					Dlaset(Upper, m-1, m-1, zero, zero, u.Off(0, 1))

					//                 Generate Q in A
					//                 (Workspace: need 2*M, prefer M + M*NB)
					if err = Dorglq(m, n, m, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					ie = itau
					itauq = ie + m
					itaup = itauq + m
					iwork = itaup + m

					//                 Bidiagonalize L in U
					//                 (Workspace: need 4*M, prefer 3*M + 2*M*NB)
					if err = Dgebrd(m, m, u, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Multiply right vectors bidiagonalizing L by Q in A
					//                 (Workspace: need 3*M + N, prefer 3*M + N*NB)
					if err = Dormbr('P', Left, Trans, m, n, m, u, work.Off(itaup-1), a, work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}

					//                 Generate left vectors bidiagonalizing L in U
					//                 (Workspace: need 4*M, prefer 3*M + M*NB)
					if err = Dorgbr('Q', m, m, m, u, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
						panic(err)
					}
					iwork = ie + m

					//                 Perform bidiagonal QR iteration, computing left
					//                 singular vectors of A in U and computing right
					//                 singular vectors of A in A
					//                 (Workspace: need BDSPAC)
					info, err = Dbdsqr(Upper, m, n, m, 0, s, work.Off(ie-1), a, u, dum.Matrix(1, opts), work.Off(iwork-1))

				}

			} else if wntvs {

				if wntun {
					//                 Path 4t(N much larger than M, JOBU='N', JOBVT='S')
					//                 M right singular vectors to be computed in VT and
					//                 no left singular vectors to be computed
					if lwork >= m*m+max(4*m, bdspac) {
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
						//                    (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
						if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy L to WORK(IR), zeroing out above it
						Dlacpy(Lower, m, m, a, work.MatrixOff(ir-1, ldwrkr, opts))
						Dlaset(Upper, m-1, m-1, zero, zero, work.MatrixOff(ir+ldwrkr-1, ldwrkr, opts))

						//                    Generate Q in A
						//                    (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
						if err = Dorglq(m, n, m, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = itau
						itauq = ie + m
						itaup = itauq + m
						iwork = itaup + m

						//                    Bidiagonalize L in WORK(IR)
						//                    (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
						if err = Dgebrd(m, m, work.MatrixOff(ir-1, ldwrkr, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right vectors bidiagonalizing L in
						//                    WORK(IR)
						//                    (Workspace: need M*M + 4*M, prefer M*M + 3*M + (M-1)*NB)
						if err = Dorgbr('P', m, m, m, work.MatrixOff(ir-1, ldwrkr, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + m

						//                    Perform bidiagonal QR iteration, computing right
						//                    singular vectors of L in WORK(IR)
						//                    (Workspace: need M*M + BDSPAC)
						info, err = Dbdsqr(Upper, m, m, 0, 0, s, work.Off(ie-1), work.MatrixOff(ir-1, ldwrkr, opts), dum.Matrix(1, opts), dum.Matrix(1, opts), work.Off(iwork-1))

						//                    Multiply right singular vectors of L in WORK(IR) by
						//                    Q in A, storing result in VT
						//                    (Workspace: need M*M)
						err = goblas.Dgemm(NoTrans, NoTrans, m, n, m, one, work.MatrixOff(ir-1, ldwrkr, opts), a, zero, vt)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + m

						//                    Compute A=L*Q
						//                    (Workspace: need 2*M, prefer M + M*NB)
						if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy result to VT
						Dlacpy(Upper, m, n, a, vt)

						//                    Generate Q in VT
						//                    (Workspace: need 2*M, prefer M + M*NB)
						if err = Dorglq(m, n, m, vt, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = itau
						itauq = ie + m
						itaup = itauq + m
						iwork = itaup + m

						//                    Zero out above L in A
						Dlaset(Upper, m-1, m-1, zero, zero, a.Off(0, 1))

						//                    Bidiagonalize L in A
						//                    (Workspace: need 4*M, prefer 3*M + 2*M*NB)
						if err = Dgebrd(m, m, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply right vectors bidiagonalizing L by Q in VT
						//                    (Workspace: need 3*M + N, prefer 3*M + N*NB)
						if err = Dormbr('P', Left, Trans, m, n, m, a, work.Off(itaup-1), vt, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + m

						//                    Perform bidiagonal QR iteration, computing right
						//                    singular vectors of A in VT
						//                    (Workspace: need BDSPAC)
						info, err = Dbdsqr(Upper, m, n, 0, 0, s, work.Off(ie-1), vt, dum.Matrix(1, opts), dum.Matrix(1, opts), work.Off(iwork-1))

					}

				} else if wntuo {
					//                 Path 5t(N much larger than M, JOBU='O', JOBVT='S')
					//                 M right singular vectors to be computed in VT and
					//                 M left singular vectors to be overwritten on A
					if lwork >= 2*m*m+max(4*m, bdspac) {
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
						//                    (Workspace: need 2*M*M + 2*M, prefer 2*M*M + M + M*NB)
						if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy L to WORK(IU), zeroing out below it
						Dlacpy(Lower, m, m, a, work.MatrixOff(iu-1, ldwrku, opts))
						Dlaset(Upper, m-1, m-1, zero, zero, work.MatrixOff(iu+ldwrku-1, ldwrku, opts))

						//                    Generate Q in A
						//                    (Workspace: need 2*M*M + 2*M, prefer 2*M*M + M + M*NB)
						if err = Dorglq(m, n, m, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = itau
						itauq = ie + m
						itaup = itauq + m
						iwork = itaup + m

						//                    Bidiagonalize L in WORK(IU), copying result to
						//                    WORK(IR)
						//                    (Workspace: need 2*M*M + 4*M,
						//                                prefer 2*M*M+3*M+2*M*NB)
						if err = Dgebrd(m, m, work.MatrixOff(iu-1, ldwrku, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Lower, m, m, work.MatrixOff(iu-1, ldwrku, opts), work.MatrixOff(ir-1, ldwrkr, opts))

						//                    Generate right bidiagonalizing vectors in WORK(IU)
						//                    (Workspace: need 2*M*M + 4*M-1,
						//                                prefer 2*M*M+3*M+(M-1)*NB)
						if err = Dorgbr('P', m, m, m, work.MatrixOff(iu-1, ldwrku, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left bidiagonalizing vectors in WORK(IR)
						//                    (Workspace: need 2*M*M + 4*M, prefer 2*M*M + 3*M + M*NB)
						if err = Dorgbr('Q', m, m, m, work.MatrixOff(ir-1, ldwrkr, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + m

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of L in WORK(IR) and computing
						//                    right singular vectors of L in WORK(IU)
						//                    (Workspace: need 2*M*M + BDSPAC)
						info, err = Dbdsqr(Upper, m, m, m, 0, s, work.Off(ie-1), work.MatrixOff(iu-1, ldwrku, opts), work.MatrixOff(ir-1, ldwrkr, opts), dum.Matrix(1, opts), work.Off(iwork-1))

						//                    Multiply right singular vectors of L in WORK(IU) by
						//                    Q in A, storing result in VT
						//                    (Workspace: need M*M)
						err = goblas.Dgemm(NoTrans, NoTrans, m, n, m, one, work.MatrixOff(iu-1, ldwrku, opts), a, zero, vt)

						//                    Copy left singular vectors of L to A
						//                    (Workspace: need M*M)
						Dlacpy(Full, m, m, work.MatrixOff(ir-1, ldwrkr, opts), a)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + m

						//                    Compute A=L*Q, copying result to VT
						//                    (Workspace: need 2*M, prefer M + M*NB)
						if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Upper, m, n, a, vt)

						//                    Generate Q in VT
						//                    (Workspace: need 2*M, prefer M + M*NB)
						if err = Dorglq(m, n, m, vt, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = itau
						itauq = ie + m
						itaup = itauq + m
						iwork = itaup + m

						//                    Zero out above L in A
						Dlaset(Upper, m-1, m-1, zero, zero, a.Off(0, 1))

						//                    Bidiagonalize L in A
						//                    (Workspace: need 4*M, prefer 3*M + 2*M*NB)
						if err = Dgebrd(m, m, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply right vectors bidiagonalizing L by Q in VT
						//                    (Workspace: need 3*M + N, prefer 3*M + N*NB)
						if err = Dormbr('P', Left, Trans, m, n, m, a, work.Off(itaup-1), vt, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left bidiagonalizing vectors of L in A
						//                    (Workspace: need 4*M, prefer 3*M + M*NB)
						if err = Dorgbr('Q', m, m, m, a, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + m

						//                    Perform bidiagonal QR iteration, compute left
						//                    singular vectors of A in A and compute right
						//                    singular vectors of A in VT
						//                    (Workspace: need BDSPAC)
						info, err = Dbdsqr(Upper, m, n, m, 0, s, work.Off(ie-1), vt, a, dum.Matrix(1, opts), work.Off(iwork-1))

					}

				} else if wntuas {
					//                 Path 6t(N much larger than M, JOBU='S' or 'A',
					//                         JOBVT='S')
					//                 M right singular vectors to be computed in VT and
					//                 M left singular vectors to be computed in U
					if lwork >= m*m+max(4*m, bdspac) {
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
						//                    (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
						if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy L to WORK(IU), zeroing out above it
						Dlacpy(Lower, m, m, a, work.MatrixOff(iu-1, ldwrku, opts))
						Dlaset(Upper, m-1, m-1, zero, zero, work.MatrixOff(iu+ldwrku-1, ldwrku, opts))

						//                    Generate Q in A
						//                    (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
						if err = Dorglq(m, n, m, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = itau
						itauq = ie + m
						itaup = itauq + m
						iwork = itaup + m

						//                    Bidiagonalize L in WORK(IU), copying result to U
						//                    (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
						if err = Dgebrd(m, m, work.MatrixOff(iu-1, ldwrku, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Lower, m, m, work.MatrixOff(iu-1, ldwrku, opts), u)

						//                    Generate right bidiagonalizing vectors in WORK(IU)
						//                    (Workspace: need M*M + 4*M-1,
						//                                prefer M*M+3*M+(M-1)*NB)
						if err = Dorgbr('P', m, m, m, work.MatrixOff(iu-1, ldwrku, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left bidiagonalizing vectors in U
						//                    (Workspace: need M*M + 4*M, prefer M*M + 3*M + M*NB)
						if err = Dorgbr('Q', m, m, m, u, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + m

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of L in U and computing right
						//                    singular vectors of L in WORK(IU)
						//                    (Workspace: need M*M + BDSPAC)
						info, err = Dbdsqr(Upper, m, m, m, 0, s, work.Off(ie-1), work.MatrixOff(iu-1, ldwrku, opts), u, dum.Matrix(1, opts), work.Off(iwork-1))

						//                    Multiply right singular vectors of L in WORK(IU) by
						//                    Q in A, storing result in VT
						//                    (Workspace: need M*M)
						err = goblas.Dgemm(NoTrans, NoTrans, m, n, m, one, work.MatrixOff(iu-1, ldwrku, opts), a, zero, vt)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + m

						//                    Compute A=L*Q, copying result to VT
						//                    (Workspace: need 2*M, prefer M + M*NB)
						if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Upper, m, n, a, vt)

						//                    Generate Q in VT
						//                    (Workspace: need 2*M, prefer M + M*NB)
						if err = Dorglq(m, n, m, vt, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy L to U, zeroing out above it
						Dlacpy(Lower, m, m, a, u)
						Dlaset(Upper, m-1, m-1, zero, zero, u.Off(0, 1))
						ie = itau
						itauq = ie + m
						itaup = itauq + m
						iwork = itaup + m

						//                    Bidiagonalize L in U
						//                    (Workspace: need 4*M, prefer 3*M + 2*M*NB)
						if err = Dgebrd(m, m, u, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply right bidiagonalizing vectors in U by Q
						//                    in VT
						//                    (Workspace: need 3*M + N, prefer 3*M + N*NB)
						if err = Dormbr('P', Left, Trans, m, n, m, u, work.Off(itaup-1), vt, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left bidiagonalizing vectors in U
						//                    (Workspace: need 4*M, prefer 3*M + M*NB)
						if err = Dorgbr('Q', m, m, m, u, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + m

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in VT
						//                    (Workspace: need BDSPAC)
						info, err = Dbdsqr(Upper, m, n, m, 0, s, work.Off(ie-1), vt, u, dum.Matrix(1, opts), work.Off(iwork-1))

					}

				}

			} else if wntva {

				if wntun {
					//                 Path 7t(N much larger than M, JOBU='N', JOBVT='A')
					//                 N right singular vectors to be computed in VT and
					//                 no left singular vectors to be computed
					if lwork >= m*m+max(n+m, 4*m, bdspac) {
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
						//                    (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
						if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Upper, m, n, a, vt)

						//                    Copy L to WORK(IR), zeroing out above it
						Dlacpy(Lower, m, m, a, work.MatrixOff(ir-1, ldwrkr, opts))
						Dlaset(Upper, m-1, m-1, zero, zero, work.MatrixOff(ir+ldwrkr-1, ldwrkr, opts))

						//                    Generate Q in VT
						//                    (Workspace: need M*M + M + N, prefer M*M + M + N*NB)
						if err = Dorglq(n, n, m, vt, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = itau
						itauq = ie + m
						itaup = itauq + m
						iwork = itaup + m

						//                    Bidiagonalize L in WORK(IR)
						//                    (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
						if err = Dgebrd(m, m, work.MatrixOff(ir-1, ldwrkr, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate right bidiagonalizing vectors in WORK(IR)
						//                    (Workspace: need M*M + 4*M-1,
						//                                prefer M*M+3*M+(M-1)*NB)
						if err = Dorgbr('P', m, m, m, work.MatrixOff(ir-1, ldwrkr, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + m

						//                    Perform bidiagonal QR iteration, computing right
						//                    singular vectors of L in WORK(IR)
						//                    (Workspace: need M*M + BDSPAC)
						info, err = Dbdsqr(Upper, m, m, 0, 0, s, work.Off(ie-1), work.MatrixOff(ir-1, ldwrkr, opts), dum.Matrix(1, opts), dum.Matrix(1, opts), work.Off(iwork-1))

						//                    Multiply right singular vectors of L in WORK(IR) by
						//                    Q in VT, storing result in A
						//                    (Workspace: need M*M)
						err = goblas.Dgemm(NoTrans, NoTrans, m, n, m, one, work.MatrixOff(ir-1, ldwrkr, opts), vt, zero, a)

						//                    Copy right singular vectors of A from A to VT
						Dlacpy(Full, m, n, a, vt)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + m

						//                    Compute A=L*Q, copying result to VT
						//                    (Workspace: need 2*M, prefer M + M*NB)
						if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Upper, m, n, a, vt)

						//                    Generate Q in VT
						//                    (Workspace: need M + N, prefer M + N*NB)
						if err = Dorglq(n, n, m, vt, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = itau
						itauq = ie + m
						itaup = itauq + m
						iwork = itaup + m

						//                    Zero out above L in A
						Dlaset(Upper, m-1, m-1, zero, zero, a.Off(0, 1))

						//                    Bidiagonalize L in A
						//                    (Workspace: need 4*M, prefer 3*M + 2*M*NB)
						if err = Dgebrd(m, m, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply right bidiagonalizing vectors in A by Q
						//                    in VT
						//                    (Workspace: need 3*M + N, prefer 3*M + N*NB)
						if err = Dormbr('P', Left, Trans, m, n, m, a, work.Off(itaup-1), vt, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + m

						//                    Perform bidiagonal QR iteration, computing right
						//                    singular vectors of A in VT
						//                    (Workspace: need BDSPAC)
						info, err = Dbdsqr(Upper, m, n, 0, 0, s, work.Off(ie-1), vt, dum.Matrix(1, opts), dum.Matrix(1, opts), work.Off(iwork-1))

					}

				} else if wntuo {
					//                 Path 8t(N much larger than M, JOBU='O', JOBVT='A')
					//                 N right singular vectors to be computed in VT and
					//                 M left singular vectors to be overwritten on A
					if lwork >= 2*m*m+max(n+m, 4*m, bdspac) {
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
						//                    (Workspace: need 2*M*M + 2*M, prefer 2*M*M + M + M*NB)
						if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Upper, m, n, a, vt)

						//                    Generate Q in VT
						//                    (Workspace: need 2*M*M + M + N, prefer 2*M*M + M + N*NB)
						if err = Dorglq(n, n, m, vt, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy L to WORK(IU), zeroing out above it
						Dlacpy(Lower, m, m, a, work.MatrixOff(iu-1, ldwrku, opts))
						Dlaset(Upper, m-1, m-1, zero, zero, work.MatrixOff(iu+ldwrku-1, ldwrku, opts))
						ie = itau
						itauq = ie + m
						itaup = itauq + m
						iwork = itaup + m

						//                    Bidiagonalize L in WORK(IU), copying result to
						//                    WORK(IR)
						//                    (Workspace: need 2*M*M + 4*M,
						//                                prefer 2*M*M+3*M+2*M*NB)
						if err = Dgebrd(m, m, work.MatrixOff(iu-1, ldwrku, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Lower, m, m, work.MatrixOff(iu-1, ldwrku, opts), work.MatrixOff(ir-1, ldwrkr, opts))

						//                    Generate right bidiagonalizing vectors in WORK(IU)
						//                    (Workspace: need 2*M*M + 4*M-1,
						//                                prefer 2*M*M+3*M+(M-1)*NB)
						if err = Dorgbr('P', m, m, m, work.MatrixOff(iu-1, ldwrku, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left bidiagonalizing vectors in WORK(IR)
						//                    (Workspace: need 2*M*M + 4*M, prefer 2*M*M + 3*M + M*NB)
						if err = Dorgbr('Q', m, m, m, work.MatrixOff(ir-1, ldwrkr, opts), work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + m

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of L in WORK(IR) and computing
						//                    right singular vectors of L in WORK(IU)
						//                    (Workspace: need 2*M*M + BDSPAC)
						info, err = Dbdsqr(Upper, m, m, m, 0, s, work.Off(ie-1), work.MatrixOff(iu-1, ldwrku, opts), work.MatrixOff(ir-1, ldwrkr, opts), dum.Matrix(1, opts), work.Off(iwork-1))

						//                    Multiply right singular vectors of L in WORK(IU) by
						//                    Q in VT, storing result in A
						//                    (Workspace: need M*M)
						err = goblas.Dgemm(NoTrans, NoTrans, m, n, m, one, work.MatrixOff(iu-1, ldwrku, opts), vt, zero, a)

						//                    Copy right singular vectors of A from A to VT
						Dlacpy(Full, m, n, a, vt)

						//                    Copy left singular vectors of A from WORK(IR) to A
						Dlacpy(Full, m, m, work.MatrixOff(ir-1, ldwrkr, opts), a)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + m

						//                    Compute A=L*Q, copying result to VT
						//                    (Workspace: need 2*M, prefer M + M*NB)
						if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Upper, m, n, a, vt)

						//                    Generate Q in VT
						//                    (Workspace: need M + N, prefer M + N*NB)
						if err = Dorglq(n, n, m, vt, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						ie = itau
						itauq = ie + m
						itaup = itauq + m
						iwork = itaup + m

						//                    Zero out above L in A
						Dlaset(Upper, m-1, m-1, zero, zero, a.Off(0, 1))

						//                    Bidiagonalize L in A
						//                    (Workspace: need 4*M, prefer 3*M + 2*M*NB)
						if err = Dgebrd(m, m, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply right bidiagonalizing vectors in A by Q
						//                    in VT
						//                    (Workspace: need 3*M + N, prefer 3*M + N*NB)
						if err = Dormbr('P', Left, Trans, m, n, m, a, work.Off(itaup-1), vt, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left bidiagonalizing vectors in A
						//                    (Workspace: need 4*M, prefer 3*M + M*NB)
						if err = Dorgbr('Q', m, m, m, a, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + m

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in A and computing right
						//                    singular vectors of A in VT
						//                    (Workspace: need BDSPAC)
						info, err = Dbdsqr(Upper, m, n, m, 0, s, work.Off(ie-1), vt, a, dum.Matrix(1, opts), work.Off(iwork-1))

					}

				} else if wntuas {
					//                 Path 9t(N much larger than M, JOBU='S' or 'A',
					//                         JOBVT='A')
					//                 N right singular vectors to be computed in VT and
					//                 M left singular vectors to be computed in U
					if lwork >= m*m+max(n+m, 4*m, bdspac) {
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
						//                    (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
						if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Upper, m, n, a, vt)

						//                    Generate Q in VT
						//                    (Workspace: need M*M + M + N, prefer M*M + M + N*NB)
						if err = Dorglq(n, n, m, vt, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy L to WORK(IU), zeroing out above it
						Dlacpy(Lower, m, m, a, work.MatrixOff(iu-1, ldwrku, opts))
						Dlaset(Upper, m-1, m-1, zero, zero, work.MatrixOff(iu+ldwrku-1, ldwrku, opts))
						ie = itau
						itauq = ie + m
						itaup = itauq + m
						iwork = itaup + m

						//                    Bidiagonalize L in WORK(IU), copying result to U
						//                    (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
						if err = Dgebrd(m, m, work.MatrixOff(iu-1, ldwrku, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Lower, m, m, work.MatrixOff(iu-1, ldwrku, opts), u)

						//                    Generate right bidiagonalizing vectors in WORK(IU)
						//                    (Workspace: need M*M + 4*M, prefer M*M + 3*M + (M-1)*NB)
						if err = Dorgbr('P', m, m, m, work.MatrixOff(iu-1, ldwrku, opts), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left bidiagonalizing vectors in U
						//                    (Workspace: need M*M + 4*M, prefer M*M + 3*M + M*NB)
						if err = Dorgbr('Q', m, m, m, u, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + m

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of L in U and computing right
						//                    singular vectors of L in WORK(IU)
						//                    (Workspace: need M*M + BDSPAC)
						info, err = Dbdsqr(Upper, m, m, m, 0, s, work.Off(ie-1), work.MatrixOff(iu-1, ldwrku, opts), u, dum.Matrix(1, opts), work.Off(iwork-1))

						//                    Multiply right singular vectors of L in WORK(IU) by
						//                    Q in VT, storing result in A
						//                    (Workspace: need M*M)
						err = goblas.Dgemm(NoTrans, NoTrans, m, n, m, one, work.MatrixOff(iu-1, ldwrku, opts), vt, zero, a)

						//                    Copy right singular vectors of A from A to VT
						Dlacpy(Full, m, n, a, vt)

					} else {
						//                    Insufficient workspace for a fast algorithm
						itau = 1
						iwork = itau + m

						//                    Compute A=L*Q, copying result to VT
						//                    (Workspace: need 2*M, prefer M + M*NB)
						if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						Dlacpy(Upper, m, n, a, vt)

						//                    Generate Q in VT
						//                    (Workspace: need M + N, prefer M + N*NB)
						if err = Dorglq(n, n, m, vt, work.Off(itau-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Copy L to U, zeroing out above it
						Dlacpy(Lower, m, m, a, u)
						Dlaset(Upper, m-1, m-1, zero, zero, u.Off(0, 1))
						ie = itau
						itauq = ie + m
						itaup = itauq + m
						iwork = itaup + m

						//                    Bidiagonalize L in U
						//                    (Workspace: need 4*M, prefer 3*M + 2*M*NB)
						if err = Dgebrd(m, m, u, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Multiply right bidiagonalizing vectors in U by Q
						//                    in VT
						//                    (Workspace: need 3*M + N, prefer 3*M + N*NB)
						if err = Dormbr('P', Left, Trans, m, n, m, u, work.Off(itaup-1), vt, work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}

						//                    Generate left bidiagonalizing vectors in U
						//                    (Workspace: need 4*M, prefer 3*M + M*NB)
						if err = Dorgbr('Q', m, m, m, u, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
							panic(err)
						}
						iwork = ie + m

						//                    Perform bidiagonal QR iteration, computing left
						//                    singular vectors of A in U and computing right
						//                    singular vectors of A in VT
						//                    (Workspace: need BDSPAC)
						info, err = Dbdsqr(Upper, m, n, m, 0, s, work.Off(ie-1), vt, u, dum.Matrix(1, opts), work.Off(iwork-1))

					}

				}

			}

		} else {
			//           N .LT. MNTHR
			//
			//           Path 10t(N greater than M, but not much larger)
			//           Reduce to bidiagonal form without LQ decomposition
			ie = 1
			itauq = ie + m
			itaup = itauq + m
			iwork = itaup + m

			//           Bidiagonalize A
			//           (Workspace: need 3*M + N, prefer 3*M + (M + N)*NB)
			if err = Dgebrd(m, n, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
				panic(err)
			}
			if wntuas {
				//              If left singular vectors desired in U, copy result to U
				//              and generate left bidiagonalizing vectors in U
				//              (Workspace: need 4*M-1, prefer 3*M + (M-1)*NB)
				Dlacpy(Lower, m, m, a, u)
				if err = Dorgbr('Q', m, m, n, u, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
			}
			if wntvas {
				//              If right singular vectors desired in VT, copy result to
				//              VT and generate right bidiagonalizing vectors in VT
				//              (Workspace: need 3*M + NRVT, prefer 3*M + NRVT*NB)
				Dlacpy(Upper, m, n, a, vt)
				if wntva {
					nrvt = n
				}
				if wntvs {
					nrvt = m
				}
				if err = Dorgbr('P', nrvt, n, m, vt, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
			}
			if wntuo {
				//              If left singular vectors desired in A, generate left
				//              bidiagonalizing vectors in A
				//              (Workspace: need 4*M-1, prefer 3*M + (M-1)*NB)
				if err = Dorgbr('Q', m, m, n, a, work.Off(itauq-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
			}
			if wntvo {
				//              If right singular vectors desired in A, generate right
				//              bidiagonalizing vectors in A
				//              (Workspace: need 4*M, prefer 3*M + M*NB)
				if err = Dorgbr('P', m, n, m, a, work.Off(itaup-1), work.Off(iwork-1), lwork-iwork+1); err != nil {
					panic(err)
				}
			}
			iwork = ie + m
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
				//              (Workspace: need BDSPAC)
				info, err = Dbdsqr(Lower, m, ncvt, nru, 0, s, work.Off(ie-1), vt, u, dum.Matrix(1, opts), work.Off(iwork-1))
			} else if (!wntuo) && wntvo {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in U and computing right singular
				//              vectors in A
				//              (Workspace: need BDSPAC)
				info, err = Dbdsqr(Lower, m, ncvt, nru, 0, s, work.Off(ie-1), a, u, dum.Matrix(1, opts), work.Off(iwork-1))
			} else {
				//              Perform bidiagonal QR iteration, if desired, computing
				//              left singular vectors in A and computing right singular
				//              vectors in VT
				//              (Workspace: need BDSPAC)
				info, err = Dbdsqr(Lower, m, ncvt, nru, 0, s, work.Off(ie-1), vt, a, dum.Matrix(1, opts), work.Off(iwork-1))
			}

		}

	}

	//     If DBDSQR failed to converge, copy unconverged superdiagonals
	//     to WORK( 2:MINMN )
	if info != 0 {
		if ie > 2 {
			for i = 1; i <= minmn-1; i++ {
				work.Set(i, work.Get(i+ie-1-1))
			}
		}
		if ie < 2 {
			for i = minmn - 1; i >= 1; i-- {
				work.Set(i, work.Get(i+ie-1-1))
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
			if err = Dlascl('G', 0, 0, bignum, anrm, minmn-1, 1, work.MatrixOff(1, minmn, opts)); err != nil {
				panic(err)
			}
		}
		if anrm < smlnum {
			if err = Dlascl('G', 0, 0, smlnum, anrm, minmn, 1, s.Matrix(minmn, opts)); err != nil {
				panic(err)
			}
		}
		if info != 0 && anrm < smlnum {
			if err = Dlascl('G', 0, 0, smlnum, anrm, minmn-1, 1, work.MatrixOff(1, minmn, opts)); err != nil {
				panic(err)
			}
		}
	}

	//     Return optimal workspace in WORK(1)
	work.Set(0, float64(maxwrk))

	return
}
