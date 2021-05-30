package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
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
//  DGESVDX uses an eigenvalue problem for obtaining the SVD, which
//  allows for the computation of a subset of singular values and
//  vectors. See DBDSVDX for details.
//
//  Note that the routine returns V**T, not V.
func Dgesvdx(jobu, jobvt, _range byte, m, n *int, a *mat.Matrix, lda *int, vl, vu *float64, il, iu, ns *int, s *mat.Vector, u *mat.Matrix, ldu *int, vt *mat.Matrix, ldvt *int, work *mat.Vector, lwork *int, iwork *[]int, info *int) {
	var alls, inds, lquery, vals, wantu, wantvt bool
	var jobz, rngtgk byte
	var anrm, bignum, eps, one, smlnum, zero float64
	var i, id, ie, ierr, ilqf, iltgk, iqrf, iscl, itau, itaup, itauq, itemp, itgkz, iutgk, j, maxwrk, minmn, minwrk, mnthr int

	dum := vf(1)

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
				mnthr = Ilaenv(toPtr(6), []byte("DGESVD"), []byte{jobu, jobvt}, m, n, toPtr(0), toPtr(0))
				if (*m) >= mnthr {
					//                 Path 1 (M much larger than N)
					maxwrk = (*n) + (*n)*Ilaenv(toPtr(1), []byte("DGEQRF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
					maxwrk = maxint(maxwrk, (*n)*((*n)+5)+2*(*n)*Ilaenv(toPtr(1), []byte("DGEBRD"), []byte{' '}, n, n, toPtr(-1), toPtr(-1)))
					if wantu {
						maxwrk = maxint(maxwrk, (*n)*((*n)*3+6)+(*n)*Ilaenv(toPtr(1), []byte("DORMQR"), []byte{' '}, n, n, toPtr(-1), toPtr(-1)))
					}
					if wantvt {
						maxwrk = maxint(maxwrk, (*n)*((*n)*3+6)+(*n)*Ilaenv(toPtr(1), []byte("DORMLQ"), []byte{' '}, n, n, toPtr(-1), toPtr(-1)))
					}
					minwrk = (*n) * ((*n)*3 + 20)
				} else {
					//                 Path 2 (M at least N, but not much larger)
					maxwrk = 4*(*n) + ((*m)+(*n))*Ilaenv(toPtr(1), []byte("DGEBRD"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
					if wantu {
						maxwrk = maxint(maxwrk, (*n)*((*n)*2+5)+(*n)*Ilaenv(toPtr(1), []byte("DORMQR"), []byte{' '}, n, n, toPtr(-1), toPtr(-1)))
					}
					if wantvt {
						maxwrk = maxint(maxwrk, (*n)*((*n)*2+5)+(*n)*Ilaenv(toPtr(1), []byte("DORMLQ"), []byte{' '}, n, n, toPtr(-1), toPtr(-1)))
					}
					minwrk = maxint((*n)*((*n)*2+19), 4*(*n)+(*m))
				}
			} else {
				mnthr = Ilaenv(toPtr(6), []byte("DGESVD"), []byte{jobu, jobvt}, m, n, toPtr(0), toPtr(0))
				if (*n) >= mnthr {
					//                 Path 1t (N much larger than M)
					maxwrk = (*m) + (*m)*Ilaenv(toPtr(1), []byte("DGELQF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
					maxwrk = maxint(maxwrk, (*m)*((*m)+5)+2*(*m)*Ilaenv(toPtr(1), []byte("DGEBRD"), []byte{' '}, m, m, toPtr(-1), toPtr(-1)))
					if wantu {
						maxwrk = maxint(maxwrk, (*m)*((*m)*3+6)+(*m)*Ilaenv(toPtr(1), []byte("DORMQR"), []byte{' '}, m, m, toPtr(-1), toPtr(-1)))
					}
					if wantvt {
						maxwrk = maxint(maxwrk, (*m)*((*m)*3+6)+(*m)*Ilaenv(toPtr(1), []byte("DORMLQ"), []byte{' '}, m, m, toPtr(-1), toPtr(-1)))
					}
					minwrk = (*m) * ((*m)*3 + 20)
				} else {
					//                 Path 2t (N at least M, but not much larger)
					maxwrk = 4*(*m) + ((*m)+(*n))*Ilaenv(toPtr(1), []byte("DGEBRD"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
					if wantu {
						maxwrk = maxint(maxwrk, (*m)*((*m)*2+5)+(*m)*Ilaenv(toPtr(1), []byte("DORMQR"), []byte{' '}, m, m, toPtr(-1), toPtr(-1)))
					}
					if wantvt {
						maxwrk = maxint(maxwrk, (*m)*((*m)*2+5)+(*m)*Ilaenv(toPtr(1), []byte("DORMLQ"), []byte{' '}, m, m, toPtr(-1), toPtr(-1)))
					}
					minwrk = maxint((*m)*((*m)*2+19), 4*(*m)+(*n))
				}
			}
		}
		maxwrk = maxint(maxwrk, minwrk)
		work.Set(0, float64(maxwrk))

		if (*lwork) < minwrk && !lquery {
			(*info) = -19
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGESVDX"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		return
	}

	//     Set singular values indices accord to RANGE.
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
	anrm = Dlange('M', m, n, a, lda, dum)
	iscl = 0
	if anrm > zero && anrm < smlnum {
		iscl = 1
		Dlascl('G', toPtr(0), toPtr(0), &anrm, &smlnum, m, n, a, lda, info)
	} else if anrm > bignum {
		iscl = 1
		Dlascl('G', toPtr(0), toPtr(0), &anrm, &bignum, m, n, a, lda, info)
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
			Dgeqrf(m, n, a, lda, work.Off(itau-1), work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)

			//           Copy R into WORK and bidiagonalize it:
			//           (Workspace: need N*N+5*N, prefer N*N+4*N+2*N*NB)
			iqrf = itemp
			id = iqrf + (*n)*(*n)
			ie = id + (*n)
			itauq = ie + (*n)
			itaup = itauq + (*n)
			itemp = itaup + (*n)
			Dlacpy('U', n, n, a, lda, work.MatrixOff(iqrf-1, *n, opts), n)
			Dlaset('L', toPtr((*n)-1), toPtr((*n)-1), &zero, &zero, work.MatrixOff(iqrf+1-1, *n, opts), n)
			Dgebrd(n, n, work.MatrixOff(iqrf-1, *n, opts), n, work.Off(id-1), work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)

			//           Solve eigenvalue problem TGK*Z=Z*S.
			//           (Workspace: need 14*N + 2*N*(N+1))
			itgkz = itemp
			itemp = itgkz + (*n)*((*n)*2+1)
			Dbdsvdx('U', jobz, rngtgk, n, work.Off(id-1), work.Off(ie-1), vl, vu, &iltgk, &iutgk, ns, s, work.MatrixOff(itgkz-1, (*n)*2, opts), toPtr((*n)*2), work.Off(itemp-1), iwork, info)

			//           If needed, compute left singular vectors.
			if wantu {
				j = itgkz
				for i = 1; i <= (*ns); i++ {
					goblas.Dcopy(n, work.Off(j-1), toPtr(1), u.Vector(0, i-1), toPtr(1))
					j = j + (*n)*2
				}
				Dlaset('A', toPtr((*m)-(*n)), ns, &zero, &zero, u.Off((*n)+1-1, 0), ldu)

				//              Call DORMBR to compute QB*UB.
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				Dormbr('Q', 'L', 'N', n, ns, n, work.MatrixOff(iqrf-1, *n, opts), n, work.Off(itauq-1), u, ldu, work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)

				//              Call DORMQR to compute Q*(QB*UB).
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				Dormqr('L', 'N', m, ns, n, a, lda, work.Off(itau-1), u, ldu, work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)
			}

			//           If needed, compute right singular vectors.
			if wantvt {
				j = itgkz + (*n)
				for i = 1; i <= (*ns); i++ {
					goblas.Dcopy(n, work.Off(j-1), toPtr(1), vt.Vector(i-1, 0), ldvt)
					j = j + (*n)*2
				}

				//              Call DORMBR to compute VB**T * PB**T
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				Dormbr('P', 'R', 'T', ns, n, n, work.MatrixOff(iqrf-1, *n, opts), n, work.Off(itaup-1), vt, ldvt, work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)
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
			ie = id + (*n)
			itauq = ie + (*n)
			itaup = itauq + (*n)
			itemp = itaup + (*n)
			Dgebrd(m, n, a, lda, work.Off(id-1), work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)

			//           Solve eigenvalue problem TGK*Z=Z*S.
			//           (Workspace: need 14*N + 2*N*(N+1))
			itgkz = itemp
			itemp = itgkz + (*n)*((*n)*2+1)
			Dbdsvdx('U', jobz, rngtgk, n, work.Off(id-1), work.Off(ie-1), vl, vu, &iltgk, &iutgk, ns, s, work.MatrixOff(itgkz-1, (*n)*2, opts), toPtr((*n)*2), work.Off(itemp-1), iwork, info)

			//           If needed, compute left singular vectors.
			if wantu {
				j = itgkz
				for i = 1; i <= (*ns); i++ {
					goblas.Dcopy(n, work.Off(j-1), toPtr(1), u.Vector(0, i-1), toPtr(1))
					j = j + (*n)*2
				}
				Dlaset('A', toPtr((*m)-(*n)), ns, &zero, &zero, u.Off((*n)+1-1, 0), ldu)

				//              Call DORMBR to compute QB*UB.
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				Dormbr('Q', 'L', 'N', m, ns, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(itemp-1), toPtr((*lwork)-itemp+1), &ierr)
			}

			//           If needed, compute right singular vectors.
			if wantvt {
				j = itgkz + (*n)
				for i = 1; i <= (*ns); i++ {
					goblas.Dcopy(n, work.Off(j-1), toPtr(1), vt.Vector(i-1, 0), ldvt)
					j = j + (*n)*2
				}

				//              Call DORMBR to compute VB**T * PB**T
				//              (Workspace in WORK( ITEMP ): need N, prefer N*NB)
				Dormbr('P', 'R', 'T', ns, n, n, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(itemp-1), toPtr((*lwork)-itemp+1), &ierr)
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
			Dgelqf(m, n, a, lda, work.Off(itau-1), work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)
			//           Copy L into WORK and bidiagonalize it:
			//           (Workspace in WORK( ITEMP ): need M*M+5*N, prefer M*M+4*M+2*M*NB)

			ilqf = itemp
			id = ilqf + (*m)*(*m)
			ie = id + (*m)
			itauq = ie + (*m)
			itaup = itauq + (*m)
			itemp = itaup + (*m)
			Dlacpy('L', m, m, a, lda, work.MatrixOff(ilqf-1, *m, opts), m)
			Dlaset('U', toPtr((*m)-1), toPtr((*m)-1), &zero, &zero, work.MatrixOff(ilqf+(*m)-1, *m, opts), m)
			Dgebrd(m, m, work.MatrixOff(ilqf-1, *m, opts), m, work.Off(id-1), work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)
			//
			//           Solve eigenvalue problem TGK*Z=Z*S.
			//           (Workspace: need 2*M*M+14*M)

			itgkz = itemp
			itemp = itgkz + (*m)*((*m)*2+1)
			Dbdsvdx('U', jobz, rngtgk, m, work.Off(id-1), work.Off(ie-1), vl, vu, &iltgk, &iutgk, ns, s, work.MatrixOff(itgkz-1, (*m)*2, opts), toPtr((*m)*2), work.Off(itemp-1), iwork, info)

			//           If needed, compute left singular vectors.
			if wantu {
				j = itgkz
				for i = 1; i <= (*ns); i++ {
					goblas.Dcopy(m, work.Off(j-1), toPtr(1), u.Vector(0, i-1), toPtr(1))
					j = j + (*m)*2
				}

				//              Call DORMBR to compute QB*UB.
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				Dormbr('Q', 'L', 'N', m, ns, m, work.MatrixOff(ilqf-1, *m, opts), m, work.Off(itauq-1), u, ldu, work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)
			}

			//           If needed, compute right singular vectors.
			if wantvt {
				j = itgkz + (*m)
				for i = 1; i <= (*ns); i++ {
					goblas.Dcopy(m, work.Off(j-1), toPtr(1), vt.Vector(i-1, 0), ldvt)
					j = j + (*m)*2
				}
				Dlaset('A', ns, toPtr((*n)-(*m)), &zero, &zero, vt.Off(0, (*m)+1-1), ldvt)

				//              Call DORMBR to compute (VB**T)*(PB**T)
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				Dormbr('P', 'R', 'T', ns, m, m, work.MatrixOff(ilqf-1, *m, opts), m, work.Off(itaup-1), vt, ldvt, work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)

				//              Call DORMLQ to compute ((VB**T)*(PB**T))*Q.
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				Dormlq('R', 'N', ns, n, m, a, lda, work.Off(itau-1), vt, ldvt, work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)
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
			ie = id + (*m)
			itauq = ie + (*m)
			itaup = itauq + (*m)
			itemp = itaup + (*m)
			Dgebrd(m, n, a, lda, work.Off(id-1), work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)

			//           Solve eigenvalue problem TGK*Z=Z*S.
			//           (Workspace: need 2*M*M+14*M)
			itgkz = itemp
			itemp = itgkz + (*m)*((*m)*2+1)
			Dbdsvdx('L', jobz, rngtgk, m, work.Off(id-1), work.Off(ie-1), vl, vu, &iltgk, &iutgk, ns, s, work.MatrixOff(itgkz-1, (*m)*2, opts), toPtr((*m)*2), work.Off(itemp-1), iwork, info)

			//           If needed, compute left singular vectors.
			if wantu {
				j = itgkz
				for i = 1; i <= (*ns); i++ {
					goblas.Dcopy(m, work.Off(j-1), toPtr(1), u.Vector(0, i-1), toPtr(1))
					j = j + (*m)*2
				}

				//              Call DORMBR to compute QB*UB.
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				Dormbr('Q', 'L', 'N', m, ns, n, a, lda, work.Off(itauq-1), u, ldu, work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)
			}

			//           If needed, compute right singular vectors.
			if wantvt {
				j = itgkz + (*m)
				for i = 1; i <= (*ns); i++ {
					goblas.Dcopy(m, work.Off(j-1), toPtr(1), vt.Vector(i-1, 0), ldvt)
					j = j + (*m)*2
				}
				Dlaset('A', ns, toPtr((*n)-(*m)), &zero, &zero, vt.Off(0, (*m)+1-1), ldvt)

				//              Call DORMBR to compute VB**T * PB**T
				//              (Workspace in WORK( ITEMP ): need M, prefer M*NB)
				Dormbr('P', 'R', 'T', ns, n, m, a, lda, work.Off(itaup-1), vt, ldvt, work.Off(itemp-1), toPtr((*lwork)-itemp+1), info)
			}
		}
	}

	//     Undo scaling if necessary
	if iscl == 1 {
		if anrm > bignum {
			Dlascl('G', toPtr(0), toPtr(0), &bignum, &anrm, &minmn, toPtr(1), s.Matrix(minmn, opts), &minmn, info)
		}
		if anrm < smlnum {
			Dlascl('G', toPtr(0), toPtr(0), &smlnum, &anrm, &minmn, toPtr(1), s.Matrix(minmn, opts), &minmn, info)
		}
	}

	//     Return optimal workspace in WORK(1)
	work.Set(0, float64(maxwrk))
}
