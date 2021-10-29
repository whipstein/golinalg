package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgesvdx computes the singular value decomposition (SVD) of a complex
//  M-by-N matrix A, optionally computing the left and/or right singular
//  vectors. The SVD is written
//
//      A = U * SIGMA * transpose(V)
//
//  where SIGMA is an M-by-N matrix which is zero except for its
//  min(m,n) diagonal elements, U is an M-by-M unitary matrix, and
//  V is an N-by-N unitary matrix.  The diagonal elements of SIGMA
//  are the singular values of A; they are real and non-negative, and
//  are returned in descending order.  The first min(m,n) columns of
//  U and V are the left and right singular vectors of A.
//
//  Zgesvdx uses an eigenvalue problem for obtaining the SVD, which
//  allows for the computation of a subset of singular values and
//  vectors. See DBDSVDX for details.
//
//  Note that the routine returns V**T, not V.
func Zgesvdx(jobu, jobvt, _range byte, m, n int, a *mat.CMatrix, vl, vu float64, il, iu, ns int, s *mat.Vector, u, vt *mat.CMatrix, work *mat.CVector, lwork int, rwork *mat.Vector, iwork *[]int) (info int, err error) {
	var alls, inds, lquery, vals, wantu, wantvt bool
	var jobz, rngtgk byte
	var czero complex128
	var anrm, bignum, eps, one, smlnum, zero float64
	var i, id, ie, ilqf, iltgk, iqrf, iscl, itau, itaup, itauq, itemp, itempr, itgkz, iutgk, j, k, maxwrk, minmn, minwrk, mnthr int

	dum := vf(1)

	czero = (0.0 + 0.0*1i)
	zero = 0.0
	one = 1.0

	//     Test the input arguments.
	ns = 0
	// abstol = 2 * Dlamch(SafeMinimum)
	lquery = (lwork == -1)
	minmn = min(m, n)
	wantu = jobu == 'V'
	wantvt = jobvt == 'V'
	if wantu || wantvt {
		jobz = 'V'
	} else {
		jobz = 'N'
	}
	alls = _range == 'A'
	vals = _range == 'V'
	inds = _range == 'I'

	if jobu != 'V' && jobu != 'N' {
		err = fmt.Errorf("jobu != 'V' && jobu != 'N': jobu='%c'", jobu)
	} else if jobvt != 'V' && jobvt != 'N' {
		err = fmt.Errorf("jobvt != 'V' && jobvt != 'N': jobvt='%c'", jobvt)
	} else if !(alls || vals || inds) {
		err = fmt.Errorf("!(alls || vals || inds): _range='%c'", _range)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if m > a.Rows {
		err = fmt.Errorf("m > a.Rows: a.Rows=%v, m=%v", a.Rows, m)
	} else if minmn > 0 {
		if vals {
			if vl < zero {
				err = fmt.Errorf("vl < zero: vl=%v", vl)
			} else if vu <= vl {
				err = fmt.Errorf("vu <= vl: vl=%v, vu=%v", vl, vu)
			}
		} else if inds {
			if il < 1 || il > max(1, minmn) {
				err = fmt.Errorf("il < 1 || il > max(1, minmn): il=%v, minmn=%v", il, minmn)
			} else if iu < min(minmn, il) || iu > minmn {
				err = fmt.Errorf("iu < min(minmn, il) || iu > minmn: il=%v, iu=%v, minmn=%v", il, iu, minmn)
			}
		}
		if err == nil {
			if wantu && u.Rows < m {
				err = fmt.Errorf("wantu && u.Rows < m: jobu='%c', u.Rows=%v, m=%v", jobu, u.Rows, m)
			} else if wantvt {
				if inds {
					if vt.Rows < iu-il+1 {
						err = fmt.Errorf("vt.Rows < iu-il+1: vt.Rows=%v, il=%v, iu=%v", vt.Rows, il, iu)
					}
				} else if vt.Rows < minmn {
					err = fmt.Errorf("vt.Rows < minmn: vt.Rows=%v, minmn=%v", vt.Rows, minmn)
				}
			}
		}
	}

	//     Compute workspace
	//     (Note: Comments in the code beginning "Workspace:" describe the
	//     minimal amount of workspace needed at that point in the code,
	//     as well as the preferred amount for good performance.
	//     NB refers to the optimal block size for the immediately
	//     following subroutine, as returned by ILAENV.)
	if err == nil {
		minwrk = 1
		maxwrk = 1
		if minmn > 0 {
			if m >= n {
				mnthr = Ilaenv(6, "Zgesvd", []byte{jobu, jobvt}, m, n, 0, 0)
				if m >= mnthr {
					//                 Path 1 (M much larger than N)
					minwrk = n * (n + 5)
					maxwrk = n + n*Ilaenv(1, "Zgeqrf", []byte{' '}, m, n, -1, -1)
					maxwrk = max(maxwrk, n*n+2*n+2*n*Ilaenv(1, "Zgebrd", []byte{' '}, n, n, -1, -1))
					if wantu || wantvt {
						maxwrk = max(maxwrk, n*n+2*n+n*Ilaenv(1, "Zunmqr", []byte("LN"), n, n, n, -1))
					}
				} else {
					//                 Path 2 (M at least N, but not much larger)
					minwrk = 3*n + m
					maxwrk = 2*n + (m+n)*Ilaenv(1, "Zgebrd", []byte{' '}, m, n, -1, -1)
					if wantu || wantvt {
						maxwrk = max(maxwrk, 2*n+n*Ilaenv(1, "Zunmqr", []byte("LN"), n, n, n, -1))
					}
				}
			} else {
				mnthr = Ilaenv(6, "Zgesvd", []byte{jobu, jobvt}, m, n, 0, 0)
				if n >= mnthr {
					//                 Path 1t (N much larger than M)
					minwrk = m * (m + 5)
					maxwrk = m + m*Ilaenv(1, "Zgelqf", []byte{' '}, m, n, -1, -1)
					maxwrk = max(maxwrk, m*m+2*m+2*m*Ilaenv(1, "Zgebrd", []byte{' '}, m, m, -1, -1))
					if wantu || wantvt {
						maxwrk = max(maxwrk, m*m+2*m+m*Ilaenv(1, "Zunmqr", []byte("LN"), m, m, m, -1))
					}
				} else {
					//                 Path 2t (N greater than M, but not much larger)
					minwrk = 3*m + n
					maxwrk = 2*m + (m+n)*Ilaenv(1, "Zgebrd", []byte{' '}, m, n, -1, -1)
					if wantu || wantvt {
						maxwrk = max(maxwrk, 2*m+m*Ilaenv(1, "Zunmqr", []byte("LN"), m, m, m, -1))
					}
				}
			}
		}
		maxwrk = max(maxwrk, minwrk)
		work.SetRe(0, float64(maxwrk))

		if lwork < minwrk && !lquery {
			err = fmt.Errorf("lwork < minwrk && !lquery: lwork=%v, minwrk=%v, lquery=%v", lwork, minwrk, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zgesvdx", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 {
		return
	}

	//     Set singular values indices accord to RANGE='A'.
	if alls {
		rngtgk = 'I'
		iltgk = 1
		iutgk = min(m, n)
	} else if inds {
		rngtgk = 'I'
		iltgk = il
		iutgk = iu
	} else {
		rngtgk = 'V'
		iltgk = 0
		iutgk = 0
	}

	//     Get machine constants
	eps = Dlamch(Precision)
	smlnum = math.Sqrt(Dlamch(SafeMinimum)) / eps
	bignum = one / smlnum

	//     Scale A if max element outside _range [SMLNUM,BIGNUM]
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
		//        more rows than columns, first reduce A using the QR
		//        decomposition.
		if m >= mnthr {
			//           Path 1 (M much larger than N):
			//           A = Q * R = Q * ( QB * B * PB**T )
			//                     = Q * ( QB * ( UB * S * VB**T ) * PB**T )
			//           U = Q * QB * UB; V**T = VB**T * PB**T
			//
			//           Compute A=Q*R
			//           (Workspace: need 2*N, prefer N+N*NB)
			itau = 1
			itemp = itau + n
			if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(itemp-1), lwork-itemp+1); err != nil {
				panic(err)
			}

			//           Copy R into WORK and bidiagonalize it:
			//           (Workspace: need N*N+3*N, prefer N*N+N+2*N*NB)
			iqrf = itemp
			itauq = itemp + n*n
			itaup = itauq + n
			itemp = itaup + n
			id = 1
			ie = id + n
			itgkz = ie + n
			Zlacpy(Upper, n, n, a, work.CMatrixOff(iqrf-1, n, opts))
			Zlaset(Lower, n-1, n-1, czero, czero, work.CMatrixOff(iqrf, n, opts))
			if err = Zgebrd(n, n, work.CMatrixOff(iqrf-1, n, opts), rwork.Off(id-1), rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(itemp-1), lwork-itemp+1); err != nil {
				panic(err)
			}
			itempr = itgkz + n*(n*2+1)

			//           Solve eigenvalue problem TGK*Z=Z*S.
			//           (Workspace: need 2*N*N+14*N)
			if info, err = Dbdsvdx(Upper, jobz, rngtgk, n, rwork.Off(id-1), rwork.Off(ie-1), vl, vu, iltgk, iutgk, ns, s, rwork.MatrixOff(itgkz-1, n*2, opts), rwork.Off(itempr-1), iwork); err != nil {
				panic(err)
			}

			//           If needed, compute left singular vectors.
			if wantu {
				k = itgkz
				for i = 1; i <= ns; i++ {
					for j = 1; j <= n; j++ {
						u.SetRe(j-1, i-1, rwork.Get(k-1))
						k = k + 1
					}
					k = k + n
				}
				Zlaset(Full, m-n, ns, czero, czero, u.Off(n, 0))

				//              Call ZUNMBR to compute QB*UB.
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				if err = Zunmbr('Q', Left, NoTrans, n, ns, n, work.CMatrixOff(iqrf-1, n, opts), work.Off(itauq-1), u, work.Off(itemp-1), lwork-itemp+1); err != nil {
					panic(err)
				}

				//              Call Zunmqr to compute Q*(QB*UB).
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				if err = Zunmqr(Left, NoTrans, m, ns, n, a, work.Off(itau-1), u, work.Off(itemp-1), lwork-itemp+1); err != nil {
					panic(err)
				}
			}

			//           If needed, compute right singular vectors.
			if wantvt {
				k = itgkz + n
				for i = 1; i <= ns; i++ {
					for j = 1; j <= n; j++ {
						vt.SetRe(i-1, j-1, rwork.Get(k-1))
						k = k + 1
					}
					k = k + n
				}

				//              Call ZUNMBR to compute VB**T * PB**T
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				if err = Zunmbr('P', Right, ConjTrans, ns, n, n, work.CMatrixOff(iqrf-1, n, opts), work.Off(itaup-1), vt, work.Off(itemp-1), lwork-itemp+1); err != nil {
					panic(err)
				}
			}
		} else {
			//           Path 2 (M at least N, but not much larger)
			//           Reduce A to bidiagonal form without QR decomposition
			//           A = QB * B * PB**T = QB * ( UB * S * VB**T ) * PB**T
			//           U = QB * UB; V**T = VB**T * PB**T
			//
			//           Bidiagonalize A
			//           (Workspace: need 2*N+M, prefer 2*N+(M+N)*NB)
			itauq = 1
			itaup = itauq + n
			itemp = itaup + n
			id = 1
			ie = id + n
			itgkz = ie + n
			if err = Zgebrd(m, n, a, rwork.Off(id-1), rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(itemp-1), lwork-itemp+1); err != nil {
				panic(err)
			}
			itempr = itgkz + n*(n*2+1)

			//           Solve eigenvalue problem TGK*Z=Z*S.
			//           (Workspace: need 2*N*N+14*N)
			if info, err = Dbdsvdx(Upper, jobz, rngtgk, n, rwork.Off(id-1), rwork.Off(ie-1), vl, vu, iltgk, iutgk, ns, s, rwork.MatrixOff(itgkz-1, n*2, opts), rwork.Off(itempr-1), iwork); err != nil {
				panic(err)
			}

			//           If needed, compute left singular vectors.
			if wantu {
				k = itgkz
				for i = 1; i <= ns; i++ {
					for j = 1; j <= n; j++ {
						u.SetRe(j-1, i-1, rwork.Get(k-1))
						k = k + 1
					}
					k = k + n
				}
				Zlaset(Full, m-n, ns, czero, czero, u.Off(n, 0))

				//              Call ZUNMBR to compute QB*UB.
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				if err = Zunmbr('Q', Left, NoTrans, m, ns, n, a, work.Off(itauq-1), u, work.Off(itemp-1), lwork-itemp+1); err != nil {
					panic(err)
				}
			}

			//           If needed, compute right singular vectors.
			if wantvt {
				k = itgkz + n
				for i = 1; i <= ns; i++ {
					for j = 1; j <= n; j++ {
						vt.SetRe(i-1, j-1, rwork.Get(k-1))
						k = k + 1
					}
					k = k + n
				}

				//              Call ZUNMBR to compute VB**T * PB**T
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				if err = Zunmbr('P', Right, ConjTrans, ns, n, n, a, work.Off(itaup-1), vt, work.Off(itemp-1), lwork-itemp+1); err != nil {
					panic(err)
				}
			}
		}
	} else {
		//        A has more columns than rows. If A has sufficiently more
		//        columns than rows, first reduce A using the LQ decomposition.
		if n >= mnthr {
			//           Path 1t (N much larger than M):
			//           A = L * Q = ( QB * B * PB**T ) * Q
			//                     = ( QB * ( UB * S * VB**T ) * PB**T ) * Q
			//           U = QB * UB ; V**T = VB**T * PB**T * Q
			//
			//           Compute A=L*Q
			//           (Workspace: need 2*M, prefer M+M*NB)
			itau = 1
			itemp = itau + m
			if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(itemp-1), lwork-itemp+1); err != nil {
				panic(err)
			}
			//           Copy L into WORK and bidiagonalize it:
			//           (Workspace in WORK( ITEMP ): need M*M+3*M, prefer M*M+M+2*M*NB)
			ilqf = itemp
			itauq = ilqf + m*m
			itaup = itauq + m
			itemp = itaup + m
			id = 1
			ie = id + m
			itgkz = ie + m
			Zlacpy(Lower, m, m, a, work.CMatrixOff(ilqf-1, m, opts))
			Zlaset(Upper, m-1, m-1, czero, czero, work.CMatrixOff(ilqf+m-1, m, opts))
			if err = Zgebrd(m, m, work.CMatrixOff(ilqf-1, m, opts), rwork.Off(id-1), rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(itemp-1), lwork-itemp+1); err != nil {
				panic(err)
			}
			itempr = itgkz + m*(m*2+1)

			//           Solve eigenvalue problem TGK*Z=Z*S.
			//           (Workspace: need 2*M*M+14*M)
			if info, err = Dbdsvdx(Upper, jobz, rngtgk, m, rwork.Off(id-1), rwork.Off(ie-1), vl, vu, iltgk, iutgk, ns, s, rwork.MatrixOff(itgkz-1, m*2, opts), rwork.Off(itempr-1), iwork); err != nil {
				panic(err)
			}

			//           If needed, compute left singular vectors.
			if wantu {
				k = itgkz
				for i = 1; i <= ns; i++ {
					for j = 1; j <= m; j++ {
						u.SetRe(j-1, i-1, rwork.Get(k-1))
						k = k + 1
					}
					k = k + m
				}

				//              Call ZUNMBR to compute QB*UB.
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				if err = Zunmbr('Q', Left, NoTrans, m, ns, m, work.CMatrixOff(ilqf-1, m, opts), work.Off(itauq-1), u, work.Off(itemp-1), lwork-itemp+1); err != nil {
					panic(err)
				}
			}

			//           If needed, compute right singular vectors.
			if wantvt {
				k = itgkz + m
				for i = 1; i <= ns; i++ {
					for j = 1; j <= m; j++ {
						vt.SetRe(i-1, j-1, rwork.Get(k-1))
						k = k + 1
					}
					k = k + m
				}
				Zlaset(Full, ns, n-m, czero, czero, vt.Off(0, m))

				//              Call ZUNMBR to compute (VB**T)*(PB**T)
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				if err = Zunmbr('P', Right, ConjTrans, ns, m, m, work.CMatrixOff(ilqf-1, m, opts), work.Off(itaup-1), vt, work.Off(itemp-1), lwork-itemp+1); err != nil {
					panic(err)
				}

				//              Call ZUNMLQ to compute ((VB**T)*(PB**T))*Q.
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				if err = Zunmlq(Right, NoTrans, ns, n, m, a, work.Off(itau-1), vt, work.Off(itemp-1), lwork-itemp+1); err != nil {
					panic(err)
				}
			}
		} else {
			//           Path 2t (N greater than M, but not much larger)
			//           Reduce to bidiagonal form without LQ decomposition
			//           A = QB * B * PB**T = QB * ( UB * S * VB**T ) * PB**T
			//           U = QB * UB; V**T = VB**T * PB**T
			//
			//           Bidiagonalize A
			//           (Workspace: need 2*M+N, prefer 2*M+(M+N)*NB)
			itauq = 1
			itaup = itauq + m
			itemp = itaup + m
			id = 1
			ie = id + m
			itgkz = ie + m
			if err = Zgebrd(m, n, a, rwork.Off(id-1), rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(itemp-1), lwork-itemp+1); err != nil {
				panic(err)
			}
			itempr = itgkz + m*(m*2+1)

			//           Solve eigenvalue problem TGK*Z=Z*S.
			//           (Workspace: need 2*M*M+14*M)
			if info, err = Dbdsvdx(Lower, jobz, rngtgk, m, rwork.Off(id-1), rwork.Off(ie-1), vl, vu, iltgk, iutgk, ns, s, rwork.MatrixOff(itgkz-1, m*2, opts), rwork.Off(itempr-1), iwork); err != nil {
				panic(err)
			}

			//           If needed, compute left singular vectors.
			if wantu {
				k = itgkz
				for i = 1; i <= ns; i++ {
					for j = 1; j <= m; j++ {
						u.SetRe(j-1, i-1, rwork.Get(k-1))
						k = k + 1
					}
					k = k + m
				}

				//              Call ZUNMBR to compute QB*UB.
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				if err = Zunmbr('Q', Left, NoTrans, m, ns, n, a, work.Off(itauq-1), u, work.Off(itemp-1), lwork-itemp+1); err != nil {
					panic(err)
				}
			}

			//           If needed, compute right singular vectors.
			if wantvt {
				k = itgkz + m
				for i = 1; i <= ns; i++ {
					for j = 1; j <= m; j++ {
						vt.SetRe(i-1, j-1, rwork.Get(k-1))
						k = k + 1
					}
					k = k + m
				}
				Zlaset(Full, ns, n-m, czero, czero, vt.Off(0, m))

				//              Call ZUNMBR to compute VB**T * PB**T
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				if err = Zunmbr('P', Right, ConjTrans, ns, n, m, a, work.Off(itaup-1), vt, work.Off(itemp-1), lwork-itemp+1); err != nil {
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
	work.SetRe(0, float64(maxwrk))

	return
}
