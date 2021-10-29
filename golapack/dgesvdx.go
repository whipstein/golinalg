package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgesvdx computes the singular value decomposition (SVD) of a real
//  M-by-N matrix A, optionally computing the left and/or right singular
//  vectors. The SVD is written
//
//      A = U * SIGMA * transpose(V)
//
//  where SIGMA is an M-by-N matrix which is zero except for its
//  min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
//  V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
//  are the singular values of A; they are real and non-negative, and
//  are returned in descending order.  The first min(m,n) columns of
//  U and V are the left and right singular vectors of A.
//
//  Dgesvdx uses an eigenvalue problem for obtaining the SVD, which
//  allows for the computation of a subset of singular values and
//  vectors. See DBDSVDX for details.
//
//  Note that the routine returns V**T, not V.
func Dgesvdx(jobu, jobvt, _range byte, m, n int, a *mat.Matrix, vl, vu float64, il, iu int, s *mat.Vector, u, vt *mat.Matrix, work *mat.Vector, lwork int, iwork *[]int) (ns, info int, err error) {
	var alls, inds, lquery, vals, wantu, wantvt bool
	var jobz, rngtgk byte
	var anrm, bignum, eps, one, smlnum, zero float64
	var i, id, ie, ilqf, iltgk, iqrf, iscl, itau, itaup, itauq, itemp, itgkz, iutgk, j, maxwrk, minmn, minwrk, mnthr int

	dum := vf(1)

	zero = 0.0
	one = 1.0

	//     Test the input arguments.
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
				err = fmt.Errorf("vl < zero: _range='%c', vl=%v", _range, vl)
			} else if vu <= vl {
				err = fmt.Errorf("vu <= vl: _range='%c', vl=%v, vu=%v", _range, vl, vu)
			}
		} else if inds {
			if il < 1 || il > max(1, minmn) {
				err = fmt.Errorf("il < 1 || il > max(1, minmn): _range='%c', il=%v, m=%v, n=%v", _range, il, m, n)
			} else if iu < min(minmn, il) || iu > minmn {
				err = fmt.Errorf("iu < min(minmn, il) || iu > minmn: _range='%c', il=%v, iu=%v, m=%v, n=%v", _range, il, iu, m, n)
			}
		}
		if err == nil {
			if wantu && u.Rows < m {
				err = fmt.Errorf("wantu && u.Rows < m: jobu='%c', u.Rows=%v, m=%v", jobu, u.Rows, m)
			} else if wantvt {
				if inds {
					if vt.Rows < iu-il+1 {
						err = fmt.Errorf("vt.Rows < iu-il+1: jobvt='%c', _range='%c', vt.Rows=%v, il=%v, iu=%v", jobvt, _range, vt.Rows, il, iu)
					}
				} else if vt.Rows < minmn {
					err = fmt.Errorf("vt.Rows < minmn: jobvt='%c', _range='%c', vt.Rows=%v, m=%v, n=%v", jobvt, _range, vt.Rows, m, n)
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
				mnthr = Ilaenv(6, "Dgesvd", []byte{jobu, jobvt}, m, n, 0, 0)
				if m >= mnthr {
					//                 Path 1 (M much larger than N)
					maxwrk = n + n*Ilaenv(1, "Dgeqrf", []byte{' '}, m, n, -1, -1)
					maxwrk = max(maxwrk, n*(n+5)+2*n*Ilaenv(1, "Dgebrd", []byte{' '}, n, n, -1, -1))
					if wantu {
						maxwrk = max(maxwrk, n*(n*3+6)+n*Ilaenv(1, "Dormqr", []byte{' '}, n, n, -1, -1))
					}
					if wantvt {
						maxwrk = max(maxwrk, n*(n*3+6)+n*Ilaenv(1, "Dormlq", []byte{' '}, n, n, -1, -1))
					}
					minwrk = n * (n*3 + 20)
				} else {
					//                 Path 2 (M at least N, but not much larger)
					maxwrk = 4*n + (m+n)*Ilaenv(1, "Dgebrd", []byte{' '}, m, n, -1, -1)
					if wantu {
						maxwrk = max(maxwrk, n*(n*2+5)+n*Ilaenv(1, "Dormqr", []byte{' '}, n, n, -1, -1))
					}
					if wantvt {
						maxwrk = max(maxwrk, n*(n*2+5)+n*Ilaenv(1, "Dormlq", []byte{' '}, n, n, -1, -1))
					}
					minwrk = max(n*(n*2+19), 4*n+m)
				}
			} else {
				mnthr = Ilaenv(6, "Dgesvd", []byte{jobu, jobvt}, m, n, 0, 0)
				if n >= mnthr {
					//                 Path 1t (N much larger than M)
					maxwrk = m + m*Ilaenv(1, "Dgelqf", []byte{' '}, m, n, -1, -1)
					maxwrk = max(maxwrk, m*(m+5)+2*m*Ilaenv(1, "Dgebrd", []byte{' '}, m, m, -1, -1))
					if wantu {
						maxwrk = max(maxwrk, m*(m*3+6)+m*Ilaenv(1, "Dormqr", []byte{' '}, m, m, -1, -1))
					}
					if wantvt {
						maxwrk = max(maxwrk, m*(m*3+6)+m*Ilaenv(1, "Dormlq", []byte{' '}, m, m, -1, -1))
					}
					minwrk = m * (m*3 + 20)
				} else {
					//                 Path 2t (N at least M, but not much larger)
					maxwrk = 4*m + (m+n)*Ilaenv(1, "Dgebrd", []byte{' '}, m, n, -1, -1)
					if wantu {
						maxwrk = max(maxwrk, m*(m*2+5)+m*Ilaenv(1, "Dormqr", []byte{' '}, m, m, -1, -1))
					}
					if wantvt {
						maxwrk = max(maxwrk, m*(m*2+5)+m*Ilaenv(1, "Dormlq", []byte{' '}, m, m, -1, -1))
					}
					minwrk = max(m*(m*2+19), 4*m+n)
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
		gltest.Xerbla2("Dgesvdx", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 {
		return
	}

	//     Set singular values indices accord to RANGE.
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
			if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(itemp-1), lwork-itemp+1); err != nil {
				panic(err)
			}

			//           Copy R into WORK and bidiagonalize it:
			//           (Workspace: need N*N+5*N, prefer N*N+4*N+2*N*NB)
			iqrf = itemp
			id = iqrf + n*n
			ie = id + n
			itauq = ie + n
			itaup = itauq + n
			itemp = itaup + n
			Dlacpy(Upper, n, n, a, work.MatrixOff(iqrf-1, n, opts))
			Dlaset(Lower, n-1, n-1, zero, zero, work.MatrixOff(iqrf, n, opts))
			if err = Dgebrd(n, n, work.MatrixOff(iqrf-1, n, opts), work.Off(id-1), work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(itemp-1), lwork-itemp+1); err != nil {
				panic(err)
			}

			//           Solve eigenvalue problem TGK*Z=Z*S.
			//           (Workspace: need 14*N + 2*N*(N+1))
			itgkz = itemp
			itemp = itgkz + n*(n*2+1)
			info, err = Dbdsvdx(Upper, jobz, rngtgk, n, work.Off(id-1), work.Off(ie-1), vl, vu, iltgk, iutgk, ns, s, work.MatrixOff(itgkz-1, n*2, opts), work.Off(itemp-1), iwork)

			//           If needed, compute left singular vectors.
			if wantu {
				j = itgkz
				for i = 1; i <= ns; i++ {
					goblas.Dcopy(n, work.Off(j-1), u.Vector(0, i-1, 1))
					j = j + n*2
				}
				Dlaset(Full, m-n, ns, zero, zero, u.Off(n, 0))

				//              Call DORMBR to compute QB*UB.
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				if err = Dormbr('Q', Left, NoTrans, n, ns, n, work.MatrixOff(iqrf-1, n, opts), work.Off(itauq-1), u, work.Off(itemp-1), lwork-itemp+1); err != nil {
					panic(err)
				}

				//              Call Dormqr to compute Q*(QB*UB).
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				if err = Dormqr(Left, NoTrans, m, ns, n, a, work.Off(itau-1), u, work.Off(itemp-1), lwork-itemp+1); err != nil {
					panic(err)
				}
			}

			//           If needed, compute right singular vectors.
			if wantvt {
				j = itgkz + n
				for i = 1; i <= ns; i++ {
					goblas.Dcopy(n, work.Off(j-1), vt.Vector(i-1, 0))
					j = j + n*2
				}

				//              Call DORMBR to compute VB**T * PB**T
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				if err = Dormbr('P', Right, Trans, ns, n, n, work.MatrixOff(iqrf-1, n, opts), work.Off(itaup-1), vt, work.Off(itemp-1), lwork-itemp+1); err != nil {
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
			//           (Workspace: need 4*N+M, prefer 4*N+(M+N)*NB)
			id = 1
			ie = id + n
			itauq = ie + n
			itaup = itauq + n
			itemp = itaup + n
			if err = Dgebrd(m, n, a, work.Off(id-1), work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(itemp-1), lwork-itemp+1); err != nil {
				panic(err)
			}

			//           Solve eigenvalue problem TGK*Z=Z*S.
			//           (Workspace: need 14*N + 2*N*(N+1))
			itgkz = itemp
			itemp = itgkz + n*(n*2+1)
			info, err = Dbdsvdx(Upper, jobz, rngtgk, n, work.Off(id-1), work.Off(ie-1), vl, vu, iltgk, iutgk, ns, s, work.MatrixOff(itgkz-1, n*2, opts), work.Off(itemp-1), iwork)

			//           If needed, compute left singular vectors.
			if wantu {
				j = itgkz
				for i = 1; i <= ns; i++ {
					goblas.Dcopy(n, work.Off(j-1), u.Vector(0, i-1, 1))
					j = j + n*2
				}
				Dlaset(Full, m-n, ns, zero, zero, u.Off(n, 0))

				//              Call DORMBR to compute QB*UB.
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				if err = Dormbr('Q', Left, NoTrans, m, ns, n, a, work.Off(itauq-1), u, work.Off(itemp-1), lwork-itemp+1); err != nil {
					panic(err)
				}
			}

			//           If needed, compute right singular vectors.
			if wantvt {
				j = itgkz + n
				for i = 1; i <= ns; i++ {
					goblas.Dcopy(n, work.Off(j-1), vt.Vector(i-1, 0))
					j = j + n*2
				}

				//              Call DORMBR to compute VB**T * PB**T
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				if err = Dormbr('P', Right, Trans, ns, n, n, a, work.Off(itaup-1), vt, work.Off(itemp-1), lwork-itemp+1); err != nil {
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
			if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(itemp-1), lwork-itemp+1); err != nil {
				panic(err)
			}
			//           Copy L into WORK and bidiagonalize it:
			//           (Workspace in WORK( ITEMP ): need M*M+5*N, prefer M*M+4*M+2*M*NB)

			ilqf = itemp
			id = ilqf + m*m
			ie = id + m
			itauq = ie + m
			itaup = itauq + m
			itemp = itaup + m
			Dlacpy(Lower, m, m, a, work.MatrixOff(ilqf-1, m, opts))
			Dlaset(Upper, m-1, m-1, zero, zero, work.MatrixOff(ilqf+m-1, m, opts))
			if err = Dgebrd(m, m, work.MatrixOff(ilqf-1, m, opts), work.Off(id-1), work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(itemp-1), lwork-itemp+1); err != nil {
				panic(err)
			}
			//
			//           Solve eigenvalue problem TGK*Z=Z*S.
			//           (Workspace: need 2*M*M+14*M)

			itgkz = itemp
			itemp = itgkz + m*(m*2+1)
			info, err = Dbdsvdx(Upper, jobz, rngtgk, m, work.Off(id-1), work.Off(ie-1), vl, vu, iltgk, iutgk, ns, s, work.MatrixOff(itgkz-1, m*2, opts), work.Off(itemp-1), iwork)

			//           If needed, compute left singular vectors.
			if wantu {
				j = itgkz
				for i = 1; i <= ns; i++ {
					goblas.Dcopy(m, work.Off(j-1), u.Vector(0, i-1, 1))
					j = j + m*2
				}

				//              Call DORMBR to compute QB*UB.
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				if err = Dormbr('Q', Left, NoTrans, m, ns, m, work.MatrixOff(ilqf-1, m, opts), work.Off(itauq-1), u, work.Off(itemp-1), lwork-itemp+1); err != nil {
					panic(err)
				}
			}

			//           If needed, compute right singular vectors.
			if wantvt {
				j = itgkz + m
				for i = 1; i <= ns; i++ {
					goblas.Dcopy(m, work.Off(j-1), vt.Vector(i-1, 0))
					j = j + m*2
				}
				Dlaset(Full, ns, n-m, zero, zero, vt.Off(0, m))

				//              Call DORMBR to compute (VB**T)*(PB**T)
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				if err = Dormbr('P', Right, Trans, ns, m, m, work.MatrixOff(ilqf-1, m, opts), work.Off(itaup-1), vt, work.Off(itemp-1), lwork-itemp+1); err != nil {
					panic(err)
				}

				//              Call Dormlq to compute ((VB**T)*(PB**T))*Q.
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				if err = Dormlq(Right, NoTrans, ns, n, m, a, work.Off(itau-1), vt, work.Off(itemp-1), lwork-itemp+1); err != nil {
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
			//           (Workspace: need 4*M+N, prefer 4*M+(M+N)*NB)
			id = 1
			ie = id + m
			itauq = ie + m
			itaup = itauq + m
			itemp = itaup + m
			if err = Dgebrd(m, n, a, work.Off(id-1), work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(itemp-1), lwork-itemp+1); err != nil {
				panic(err)
			}

			//           Solve eigenvalue problem TGK*Z=Z*S.
			//           (Workspace: need 2*M*M+14*M)
			itgkz = itemp
			itemp = itgkz + m*(m*2+1)
			info, err = Dbdsvdx(Lower, jobz, rngtgk, m, work.Off(id-1), work.Off(ie-1), vl, vu, iltgk, iutgk, ns, s, work.MatrixOff(itgkz-1, m*2, opts), work.Off(itemp-1), iwork)

			//           If needed, compute left singular vectors.
			if wantu {
				j = itgkz
				for i = 1; i <= ns; i++ {
					goblas.Dcopy(m, work.Off(j-1), u.Vector(0, i-1, 1))
					j = j + m*2
				}

				//              Call DORMBR to compute QB*UB.
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				if err = Dormbr('Q', Left, NoTrans, m, ns, n, a, work.Off(itauq-1), u, work.Off(itemp-1), lwork-itemp+1); err != nil {
					panic(err)
				}
			}

			//           If needed, compute right singular vectors.
			if wantvt {
				j = itgkz + m
				for i = 1; i <= ns; i++ {
					goblas.Dcopy(m, work.Off(j-1), vt.Vector(i-1, 0))
					j = j + m*2
				}
				Dlaset(Full, ns, n-m, zero, zero, vt.Off(0, m))

				//              Call DORMBR to compute VB**T * PB**T
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				if err = Dormbr('P', Right, Trans, ns, n, m, a, work.Off(itaup-1), vt, work.Off(itemp-1), lwork-itemp+1); err != nil {
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
