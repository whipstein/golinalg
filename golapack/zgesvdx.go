package golapack

import (
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
//  minint(m,n) diagonal elements, U is an M-by-M unitary matrix, and
//  V is an N-by-N unitary matrix.  The diagonal elements of SIGMA
//  are the singular values of A; they are real and non-negative, and
//  are returned in descending order.  The first minint(m,n) columns of
//  U and V are the left and right singular vectors of A.
//
//  ZGESVDX uses an eigenvalue problem for obtaining the SVD, which
//  allows for the computation of a subset of singular values and
//  vectors. See DBDSVDX for details.
//
//  Note that the routine returns V**T, not V.
func Zgesvdx(jobu, jobvt, _range byte, m, n *int, a *mat.CMatrix, lda *int, vl, vu *float64, il, iu, ns *int, s *mat.Vector, u *mat.CMatrix, ldu *int, vt *mat.CMatrix, ldvt *int, work *mat.CVector, lwork *int, rwork *mat.Vector, iwork *[]int, info *int) {
	var alls, inds, lquery, vals, wantu, wantvt bool
	var jobz, rngtgk byte
	var czero complex128
	var anrm, bignum, eps, one, smlnum, zero float64
	var i, id, ie, ierr, ilqf, iltgk, iqrf, iscl, itau, itaup, itauq, itemp, itempr, itgkz, iutgk, j, k, maxwrk, minmn, minwrk, mnthr int
	dum := vf(1)

	czero = (0.0 + 0.0*1i)
	zero = 0.0
	one = 1.0

	//     Test the input arguments.
	(*ns) = 0
	(*info) = 0
	// abstol = 2 * Dlamch(SafeMinimum)
	lquery = ((*lwork) == -1)
	minmn = minint(*m, *n)
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

	(*info) = 0
	if jobu != 'V' && jobu != 'N' {
		(*info) = -1
	} else if jobvt != 'V' && jobvt != 'N' {
		(*info) = -2
	} else if !(alls || vals || inds) {
		(*info) = -3
	} else if (*m) < 0 {
		(*info) = -4
	} else if (*n) < 0 {
		(*info) = -5
	} else if (*m) > (*lda) {
		(*info) = -7
	} else if minmn > 0 {
		if vals {
			if (*vl) < zero {
				(*info) = -8
			} else if (*vu) <= (*vl) {
				(*info) = -9
			}
		} else if inds {
			if (*il) < 1 || (*il) > maxint(1, minmn) {
				(*info) = -10
			} else if (*iu) < minint(minmn, *il) || (*iu) > minmn {
				(*info) = -11
			}
		}
		if (*info) == 0 {
			if wantu && (*ldu) < (*m) {
				(*info) = -15
			} else if wantvt {
				if inds {
					if (*ldvt) < (*iu)-(*il)+1 {
						(*info) = -17
					}
				} else if (*ldvt) < minmn {
					(*info) = -17
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
	if (*info) == 0 {
		minwrk = 1
		maxwrk = 1
		if minmn > 0 {
			if (*m) >= (*n) {
				mnthr = Ilaenv(func() *int { y := 6; return &y }(), []byte("ZGESVD"), []byte{jobu, jobvt}, m, n, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }())
				if (*m) >= mnthr {
					//                 Path 1 (M much larger than N)
					minwrk = (*n) * ((*n) + 5)
					maxwrk = (*n) + (*n)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEQRF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
					maxwrk = maxint(maxwrk, (*n)*(*n)+2*(*n)+2*(*n)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEBRD"), []byte{' '}, n, n, toPtr(-1), toPtr(-1)))
					if wantu || wantvt {
						maxwrk = maxint(maxwrk, (*n)*(*n)+2*(*n)+(*n)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMQR"), []byte("LN"), n, n, n, toPtr(-1)))
					}
				} else {
					//                 Path 2 (M at least N, but not much larger)
					minwrk = 3*(*n) + (*m)
					maxwrk = 2*(*n) + ((*m)+(*n))*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEBRD"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
					if wantu || wantvt {
						maxwrk = maxint(maxwrk, 2*(*n)+(*n)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMQR"), []byte("LN"), n, n, n, toPtr(-1)))
					}
				}
			} else {
				mnthr = Ilaenv(func() *int { y := 6; return &y }(), []byte("ZGESVD"), []byte{jobu, jobvt}, m, n, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }())
				if (*n) >= mnthr {
					//                 Path 1t (N much larger than M)
					minwrk = (*m) * ((*m) + 5)
					maxwrk = (*m) + (*m)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGELQF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
					maxwrk = maxint(maxwrk, (*m)*(*m)+2*(*m)+2*(*m)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEBRD"), []byte{' '}, m, m, toPtr(-1), toPtr(-1)))
					if wantu || wantvt {
						maxwrk = maxint(maxwrk, (*m)*(*m)+2*(*m)+(*m)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMQR"), []byte("LN"), m, m, m, toPtr(-1)))
					}
				} else {
					//                 Path 2t (N greater than M, but not much larger)
					minwrk = 3*(*m) + (*n)
					maxwrk = 2*(*m) + ((*m)+(*n))*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEBRD"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
					if wantu || wantvt {
						maxwrk = maxint(maxwrk, 2*(*m)+(*m)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMQR"), []byte("LN"), m, m, m, toPtr(-1)))
					}
				}
			}
		}
		maxwrk = maxint(maxwrk, minwrk)
		work.SetRe(0, float64(maxwrk))

		if (*lwork) < minwrk && !lquery {
			(*info) = -19
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGESVDX"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		return
	}

	//     Set singular values indices accord to RANGE='A'.
	if alls {
		rngtgk = 'I'
		iltgk = 1
		iutgk = minint(*m, *n)
	} else if inds {
		rngtgk = 'I'
		iltgk = (*il)
		iutgk = (*iu)
	} else {
		rngtgk = 'V'
		iltgk = 0
		iutgk = 0
	}

	//     Get machine constants
	eps = Dlamch(Precision)
	smlnum = math.Sqrt(Dlamch(SafeMinimum)) / eps
	bignum = one / smlnum

	//     Scale A if maxint element outside _range [SMLNUM,BIGNUM]
	anrm = Zlange('M', m, n, a, lda, dum)
	iscl = 0
	if anrm > zero && anrm < smlnum {
		iscl = 1
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &smlnum, m, n, a, lda, info)
	} else if anrm > bignum {
		iscl = 1
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &bignum, m, n, a, lda, info)
	}

	if (*m) >= (*n) {
		//        A has at least as many rows as columns. If A has sufficiently
		//        more rows than columns, first reduce A using the QR
		//        decomposition.
		if (*m) >= mnthr {
			//           Path 1 (M much larger than N):
			//           A = Q * R = Q * ( QB * B * PB**T )
			//                     = Q * ( QB * ( UB * S * VB**T ) * PB**T )
			//           U = Q * QB * UB; V**T = VB**T * PB**T
			//
			//           Compute A=Q*R
			//           (Workspace: need 2*N, prefer N+N*NB)
			itau = 1
			itemp = itau + (*n)
			Zgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)

			//           Copy R into WORK and bidiagonalize it:
			//           (Workspace: need N*N+3*N, prefer N*N+N+2*N*NB)
			iqrf = itemp
			itauq = itemp + (*n)*(*n)
			itaup = itauq + (*n)
			itemp = itaup + (*n)
			id = 1
			ie = id + (*n)
			itgkz = ie + (*n)
			Zlacpy('U', n, n, a, lda, work.CMatrixOff(iqrf-1, *n, opts), n)
			Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, work.CMatrixOff(iqrf+1-1, *n, opts), n)
			Zgebrd(n, n, work.CMatrixOff(iqrf-1, *n, opts), n, rwork.Off(id-1), rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)
			itempr = itgkz + (*n)*((*n)*2+1)

			//           Solve eigenvalue problem TGK*Z=Z*S.
			//           (Workspace: need 2*N*N+14*N)
			Dbdsvdx('U', jobz, rngtgk, n, rwork.Off(id-1), rwork.Off(ie-1), vl, vu, &iltgk, &iutgk, ns, s, rwork.MatrixOff(itgkz-1, (*n)*2, opts), toPtr((*n)*2), rwork.Off(itempr-1), iwork, info)

			//           If needed, compute left singular vectors.
			if wantu {
				k = itgkz
				for i = 1; i <= (*ns); i++ {
					for j = 1; j <= (*n); j++ {
						u.SetRe(j-1, i-1, rwork.Get(k-1))
						k = k + 1
					}
					k = k + (*n)
				}
				Zlaset('A', toPtr((*m)-(*n)), ns, &czero, &czero, u.Off((*n)+1-1, 0), ldu)

				//              Call ZUNMBR to compute QB*UB.
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				Zunmbr('Q', 'L', 'N', n, ns, n, work.CMatrixOff(iqrf-1, *n, opts), n, work.Off(itauq-1), u, ldu, work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)

				//              Call ZUNMQR to compute Q*(QB*UB).
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				Zunmqr('L', 'N', m, ns, n, a, lda, work.Off(itau-1), u, ldu, work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)
			}

			//           If needed, compute right singular vectors.
			if wantvt {
				k = itgkz + (*n)
				for i = 1; i <= (*ns); i++ {
					for j = 1; j <= (*n); j++ {
						vt.SetRe(i-1, j-1, rwork.Get(k-1))
						k = k + 1
					}
					k = k + (*n)
				}

				//              Call ZUNMBR to compute VB**T * PB**T
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				Zunmbr('P', 'R', 'C', ns, n, n, work.CMatrixOff(iqrf-1, *n, opts), n, work.Off(itaup-1), vt, ldvt, work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)
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
			itaup = itauq + (*n)
			itemp = itaup + (*n)
			id = 1
			ie = id + (*n)
			itgkz = ie + (*n)
			Zgebrd(m, n, a, lda, rwork.Off(id-1), rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)
			itempr = itgkz + (*n)*((*n)*2+1)

			//           Solve eigenvalue problem TGK*Z=Z*S.
			//           (Workspace: need 2*N*N+14*N)
			Dbdsvdx('U', jobz, rngtgk, n, rwork.Off(id-1), rwork.Off(ie-1), vl, vu, &iltgk, &iutgk, ns, s, rwork.MatrixOff(itgkz-1, (*n)*2, opts), toPtr((*n)*2), rwork.Off(itempr-1), iwork, info)

			//           If needed, compute left singular vectors.
			if wantu {
				k = itgkz
				for i = 1; i <= (*ns); i++ {
					for j = 1; j <= (*n); j++ {
						u.SetRe(j-1, i-1, rwork.Get(k-1))
						k = k + 1
					}
					k = k + (*n)
				}
				Zlaset('A', toPtr((*m)-(*n)), ns, &czero, &czero, u.Off((*n)+1-1, 0), ldu)

				//              Call ZUNMBR to compute QB*UB.
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				Zunmbr('Q', 'L', 'N', m, ns, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(itemp-1), toPtr((*lwork)-itemp+1), &ierr)
			}

			//           If needed, compute right singular vectors.
			if wantvt {
				k = itgkz + (*n)
				for i = 1; i <= (*ns); i++ {
					for j = 1; j <= (*n); j++ {
						vt.SetRe(i-1, j-1, rwork.Get(k-1))
						k = k + 1
					}
					k = k + (*n)
				}

				//              Call ZUNMBR to compute VB**T * PB**T
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				Zunmbr('P', 'R', 'C', ns, n, n, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(itemp-1), toPtr((*lwork)-itemp+1), &ierr)
			}
		}
	} else {
		//        A has more columns than rows. If A has sufficiently more
		//        columns than rows, first reduce A using the LQ decomposition.
		if (*n) >= mnthr {
			//           Path 1t (N much larger than M):
			//           A = L * Q = ( QB * B * PB**T ) * Q
			//                     = ( QB * ( UB * S * VB**T ) * PB**T ) * Q
			//           U = QB * UB ; V**T = VB**T * PB**T * Q
			//
			//           Compute A=L*Q
			//           (Workspace: need 2*M, prefer M+M*NB)
			itau = 1
			itemp = itau + (*m)
			Zgelqf(m, n, a, lda, work.Off(itau-1), work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)
			//           Copy L into WORK and bidiagonalize it:
			//           (Workspace in WORK( ITEMP ): need M*M+3*M, prefer M*M+M+2*M*NB)
			ilqf = itemp
			itauq = ilqf + (*m)*(*m)
			itaup = itauq + (*m)
			itemp = itaup + (*m)
			id = 1
			ie = id + (*m)
			itgkz = ie + (*m)
			Zlacpy('L', m, m, a, lda, work.CMatrixOff(ilqf-1, *m, opts), m)
			Zlaset('U', toPtr((*m)-1), toPtr((*m)-1), &czero, &czero, work.CMatrixOff(ilqf+(*m)-1, *m, opts), m)
			Zgebrd(m, m, work.CMatrixOff(ilqf-1, *m, opts), m, rwork.Off(id-1), rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)
			itempr = itgkz + (*m)*((*m)*2+1)

			//           Solve eigenvalue problem TGK*Z=Z*S.
			//           (Workspace: need 2*M*M+14*M)
			Dbdsvdx('U', jobz, rngtgk, m, rwork.Off(id-1), rwork.Off(ie-1), vl, vu, &iltgk, &iutgk, ns, s, rwork.MatrixOff(itgkz-1, (*m)*2, opts), toPtr((*m)*2), rwork.Off(itempr-1), iwork, info)

			//           If needed, compute left singular vectors.
			if wantu {
				k = itgkz
				for i = 1; i <= (*ns); i++ {
					for j = 1; j <= (*m); j++ {
						u.SetRe(j-1, i-1, rwork.Get(k-1))
						k = k + 1
					}
					k = k + (*m)
				}

				//              Call ZUNMBR to compute QB*UB.
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				Zunmbr('Q', 'L', 'N', m, ns, m, work.CMatrixOff(ilqf-1, *m, opts), m, work.Off(itauq-1), u, ldu, work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)
			}

			//           If needed, compute right singular vectors.
			if wantvt {
				k = itgkz + (*m)
				for i = 1; i <= (*ns); i++ {
					for j = 1; j <= (*m); j++ {
						vt.SetRe(i-1, j-1, rwork.Get(k-1))
						k = k + 1
					}
					k = k + (*m)
				}
				Zlaset('A', ns, toPtr((*n)-(*m)), &czero, &czero, vt.Off(0, (*m)+1-1), ldvt)

				//              Call ZUNMBR to compute (VB**T)*(PB**T)
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				Zunmbr('P', 'R', 'C', ns, m, m, work.CMatrixOff(ilqf-1, *m, opts), m, work.Off(itaup-1), vt, ldvt, work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)

				//              Call ZUNMLQ to compute ((VB**T)*(PB**T))*Q.
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				Zunmlq('R', 'N', ns, n, m, a, lda, work.Off(itau-1), vt, ldvt, work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)
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
			itaup = itauq + (*m)
			itemp = itaup + (*m)
			id = 1
			ie = id + (*m)
			itgkz = ie + (*m)
			Zgebrd(m, n, a, lda, rwork.Off(id-1), rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)
			itempr = itgkz + (*m)*((*m)*2+1)

			//           Solve eigenvalue problem TGK*Z=Z*S.
			//           (Workspace: need 2*M*M+14*M)
			Dbdsvdx('L', jobz, rngtgk, m, rwork.Off(id-1), rwork.Off(ie-1), vl, vu, &iltgk, &iutgk, ns, s, rwork.MatrixOff(itgkz-1, (*m)*2, opts), toPtr((*m)*2), rwork.Off(itempr-1), iwork, info)

			//           If needed, compute left singular vectors.
			if wantu {
				k = itgkz
				for i = 1; i <= (*ns); i++ {
					for j = 1; j <= (*m); j++ {
						u.SetRe(j-1, i-1, rwork.Get(k-1))
						k = k + 1
					}
					k = k + (*m)
				}

				//              Call ZUNMBR to compute QB*UB.
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				Zunmbr('Q', 'L', 'N', m, ns, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)
			}

			//           If needed, compute right singular vectors.
			if wantvt {
				k = itgkz + (*m)
				for i = 1; i <= (*ns); i++ {
					for j = 1; j <= (*m); j++ {
						vt.SetRe(i-1, j-1, rwork.Get(k-1))
						k = k + 1
					}
					k = k + (*m)
				}
				Zlaset('A', ns, toPtr((*n)-(*m)), &czero, &czero, vt.Off(0, (*m)+1-1), ldvt)

				//              Call ZUNMBR to compute VB**T * PB**T
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				Zunmbr('P', 'R', 'C', ns, n, m, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)
			}
		}
	}

	//     Undo scaling if necessary
	if iscl == 1 {
		if anrm > bignum {
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bignum, &anrm, &minmn, func() *int { y := 1; return &y }(), s.Matrix(minmn, opts), &minmn, info)
		}
		if anrm < smlnum {
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &smlnum, &anrm, &minmn, func() *int { y := 1; return &y }(), s.Matrix(minmn, opts), &minmn, info)
		}
	}

	//     Return optimal workspace in WORK(1)
	work.SetRe(0, float64(maxwrk))
}
