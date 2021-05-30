package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Zggevx computes for a pair of N-by-N complex nonsymmetric matrices
// (A,B) the generalized eigenvalues, and optionally, the left and/or
// right generalized eigenvectors.
//
// Optionally, it also computes a balancing transformation to improve
// the conditioning of the eigenvalues and eigenvectors (ILO, IHI,
// LSCALE, RSCALE, ABNRM, and BBNRM), reciprocal condition numbers for
// the eigenvalues (RCONDE), and reciprocal condition numbers for the
// right eigenvectors (RCONDV).
//
// A generalized eigenvalue for a pair of matrices (A,B) is a scalar
// lambda or a ratio alpha/beta = lambda, such that A - lambda*B is
// singular. It is usually represented as the pair (alpha,beta), as
// there is a reasonable interpretation for beta=0, and even for both
// being zero.
//
// The right eigenvector v(j) corresponding to the eigenvalue lambda(j)
// of (A,B) satisfies
//                  A * v(j) = lambda(j) * B * v(j) .
// The left eigenvector u(j) corresponding to the eigenvalue lambda(j)
// of (A,B) satisfies
//                  u(j)**H * A  = lambda(j) * u(j)**H * B.
// where u(j)**H is the conjugate-transpose of u(j).
func Zggevx(balanc, jobvl, jobvr, sense byte, n *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, alpha, beta *mat.CVector, vl *mat.CMatrix, ldvl *int, vr *mat.CMatrix, ldvr, ilo, ihi *int, lscale, rscale *mat.Vector, abnrm, bbnrm *float64, rconde, rcondv *mat.Vector, work *mat.CVector, lwork *int, rwork *mat.Vector, iwork *[]int, bwork *[]bool, info *int) {
	var ilascl, ilbscl, ilv, ilvl, ilvr, lquery, noscl, wantsb, wantse, wantsn, wantsv bool
	var chtemp byte
	var cone, czero complex128
	var anrm, anrmto, bignum, bnrm, bnrmto, eps, one, smlnum, temp, zero float64
	var i, icols, ierr, ijobvl, ijobvr, in, irows, itau, iwrk, iwrk1, j, jc, jr, m, maxwrk, minwrk int
	ldumma := make([]bool, 1)

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Decode the input arguments
	if jobvl == 'N' {
		ijobvl = 1
		ilvl = false
	} else if jobvl == 'V' {
		ijobvl = 2
		ilvl = true
	} else {
		ijobvl = -1
		ilvl = false
	}

	if jobvr == 'N' {
		ijobvr = 1
		ilvr = false
	} else if jobvr == 'V' {
		ijobvr = 2
		ilvr = true
	} else {
		ijobvr = -1
		ilvr = false
	}
	ilv = ilvl || ilvr

	noscl = balanc == 'N' || balanc == 'P'
	wantsn = sense == 'N'
	wantse = sense == 'E'
	wantsv = sense == 'V'
	wantsb = sense == 'B'

	//     Test the input arguments
	(*info) = 0
	lquery = ((*lwork) == -1)
	if !(noscl || balanc == 'S' || balanc == 'B') {
		(*info) = -1
	} else if ijobvl <= 0 {
		(*info) = -2
	} else if ijobvr <= 0 {
		(*info) = -3
	} else if !(wantsn || wantse || wantsb || wantsv) {
		(*info) = -4
	} else if (*n) < 0 {
		(*info) = -5
	} else if (*lda) < maxint(1, *n) {
		(*info) = -7
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -9
	} else if (*ldvl) < 1 || (ilvl && (*ldvl) < (*n)) {
		(*info) = -13
	} else if (*ldvr) < 1 || (ilvr && (*ldvr) < (*n)) {
		(*info) = -15
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       NB refers to the optimal block size for the immediately
	//       following subroutine, as returned by ILAENV. The workspace is
	//       computed assuming ILO = 1 and IHI = N, the worst case.)
	if (*info) == 0 {
		if (*n) == 0 {
			minwrk = 1
			maxwrk = 1
		} else {
			minwrk = 2 * (*n)
			if wantse {
				minwrk = 4 * (*n)
			} else if wantsv || wantsb {
				minwrk = 2 * (*n) * ((*n) + 1)
			}
			maxwrk = minwrk
			maxwrk = maxint(maxwrk, (*n)+(*n)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEQRF"), []byte{' '}, n, func() *int { y := 1; return &y }(), n, func() *int { y := 0; return &y }()))
			maxwrk = maxint(maxwrk, (*n)+(*n)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMQR"), []byte{' '}, n, func() *int { y := 1; return &y }(), n, func() *int { y := 0; return &y }()))
			if ilvl {
				maxwrk = maxint(maxwrk, (*n)+(*n)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNGQR"), []byte{' '}, n, func() *int { y := 1; return &y }(), n, func() *int { y := 0; return &y }()))
			}
		}
		work.SetRe(0, float64(maxwrk))

		if (*lwork) < minwrk && !lquery {
			(*info) = -25
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGGEVX"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Get machine constants
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum)
	bignum = one / smlnum
	Dlabad(&smlnum, &bignum)
	smlnum = math.Sqrt(smlnum) / eps
	bignum = one / smlnum

	//     Scale A if maxint element outside range [SMLNUM,BIGNUM]
	anrm = Zlange('M', n, n, a, lda, rwork)
	ilascl = false
	if anrm > zero && anrm < smlnum {
		anrmto = smlnum
		ilascl = true
	} else if anrm > bignum {
		anrmto = bignum
		ilascl = true
	}
	if ilascl {
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &anrmto, n, n, a, lda, &ierr)
	}

	//     Scale B if maxint element outside range [SMLNUM,BIGNUM]
	bnrm = Zlange('M', n, n, b, ldb, rwork)
	ilbscl = false
	if bnrm > zero && bnrm < smlnum {
		bnrmto = smlnum
		ilbscl = true
	} else if bnrm > bignum {
		bnrmto = bignum
		ilbscl = true
	}
	if ilbscl {
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrm, &bnrmto, n, n, b, ldb, &ierr)
	}

	//     Permute and/or balance the matrix pair (A,B)
	//     (Real Workspace: need 6*N if BALANC = 'S' or 'B', 1 otherwise)
	Zggbal(balanc, n, a, lda, b, ldb, ilo, ihi, lscale, rscale, rwork, &ierr)

	//     Compute ABNRM and BBNRM
	(*abnrm) = Zlange('1', n, n, a, lda, rwork.Off(0))
	if ilascl {
		rwork.Set(0, (*abnrm))
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrmto, &anrm, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), rwork.Matrix(1, opts), func() *int { y := 1; return &y }(), &ierr)
		(*abnrm) = rwork.Get(0)
	}

	(*bbnrm) = Zlange('1', n, n, b, ldb, rwork.Off(0))
	if ilbscl {
		rwork.Set(0, (*bbnrm))
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrmto, &bnrm, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), rwork.Matrix(1, opts), func() *int { y := 1; return &y }(), &ierr)
		(*bbnrm) = rwork.Get(0)
	}

	//     Reduce B to triangular form (QR decomposition of B)
	//     (Complex Workspace: need N, prefer N*NB )
	irows = (*ihi) + 1 - (*ilo)
	if ilv || !wantsn {
		icols = (*n) + 1 - (*ilo)
	} else {
		icols = irows
	}
	itau = 1
	iwrk = itau + irows
	Zgeqrf(&irows, &icols, b.Off((*ilo)-1, (*ilo)-1), ldb, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)

	//     Apply the unitary transformation to A
	//     (Complex Workspace: need N, prefer N*NB)
	Zunmqr('L', 'C', &irows, &icols, &irows, b.Off((*ilo)-1, (*ilo)-1), ldb, work.Off(itau-1), a.Off((*ilo)-1, (*ilo)-1), lda, work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)

	//     Initialize VL and/or VR
	//     (Workspace: need N, prefer N*NB)
	if ilvl {
		Zlaset('F', n, n, &czero, &cone, vl, ldvl)
		if irows > 1 {
			Zlacpy('L', toPtr(irows-1), toPtr(irows-1), b.Off((*ilo)+1-1, (*ilo)-1), ldb, vl.Off((*ilo)+1-1, (*ilo)-1), ldvl)
		}
		Zungqr(&irows, &irows, &irows, vl.Off((*ilo)-1, (*ilo)-1), ldvl, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)
	}

	if ilvr {
		Zlaset('F', n, n, &czero, &cone, vr, ldvr)
	}

	//     Reduce to generalized Hessenberg form
	//     (Workspace: none needed)
	if ilv || !wantsn {
		//        Eigenvectors requested -- work on whole matrix.
		Zgghrd(jobvl, jobvr, n, ilo, ihi, a, lda, b, ldb, vl, ldvl, vr, ldvr, &ierr)
	} else {
		Zgghrd('N', 'N', &irows, func() *int { y := 1; return &y }(), &irows, a.Off((*ilo)-1, (*ilo)-1), lda, b.Off((*ilo)-1, (*ilo)-1), ldb, vl, ldvl, vr, ldvr, &ierr)
	}

	//     Perform QZ algorithm (Compute eigenvalues, and optionally, the
	//     Schur forms and Schur vectors)
	//     (Complex Workspace: need N)
	//     (Real Workspace: need N)
	iwrk = itau
	if ilv || !wantsn {
		chtemp = 'S'
	} else {
		chtemp = 'E'
	}

	Zhgeqz(chtemp, jobvl, jobvr, n, ilo, ihi, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr, work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), rwork, &ierr)
	if ierr != 0 {
		if ierr > 0 && ierr <= (*n) {
			(*info) = ierr
		} else if ierr > (*n) && ierr <= 2*(*n) {
			(*info) = ierr - (*n)
		} else {
			(*info) = (*n) + 1
		}
		goto label90
	}

	//     Compute Eigenvectors and estimate condition numbers if desired
	//     ZTGEVC: (Complex Workspace: need 2*N )
	//             (Real Workspace:    need 2*N )
	//     ZTGSNA: (Complex Workspace: need 2*N*N if SENSE='V' or 'B')
	//             (Integer Workspace: need N+2 )
	if ilv || !wantsn {
		if ilv {
			if ilvl {
				if ilvr {
					chtemp = 'B'
				} else {
					chtemp = 'L'
				}
			} else {
				chtemp = 'R'
			}

			Ztgevc(chtemp, 'B', ldumma, n, a, lda, b, ldb, vl, ldvl, vr, ldvr, n, &in, work.Off(iwrk-1), rwork, &ierr)
			if ierr != 0 {
				(*info) = (*n) + 2
				goto label90
			}
		}

		if !wantsn {
			//           compute eigenvectors (DTGEVC) and estimate condition
			//           numbers (DTGSNA). Note that the definition of the condition
			//           number is not invariant under transformation (u,v) to
			//           (Q*u, Z*v), where (u,v) are eigenvectors of the generalized
			//           Schur form (S,T), Q and Z are orthogonal matrices. In order
			//           to avoid using extra 2*N*N workspace, we have to
			//           re-calculate eigenvectors and estimate the condition numbers
			//           one at a time.
			for i = 1; i <= (*n); i++ {

				for j = 1; j <= (*n); j++ {
					(*bwork)[j-1] = false
				}
				(*bwork)[i-1] = true

				iwrk = (*n) + 1
				iwrk1 = iwrk + (*n)

				if wantse || wantsb {
					Ztgevc('B', 'S', *bwork, n, a, lda, b, ldb, work.CMatrix(*n, opts), n, work.CMatrixOff(iwrk-1, *n, opts), n, func() *int { y := 1; return &y }(), &m, work.Off(iwrk1-1), rwork, &ierr)
					if ierr != 0 {
						(*info) = (*n) + 2
						goto label90
					}
				}

				Ztgsna(sense, 'S', *bwork, n, a, lda, b, ldb, work.CMatrix(*n, opts), n, work.CMatrixOff(iwrk-1, *n, opts), n, rconde.Off(i-1), rcondv.Off(i-1), func() *int { y := 1; return &y }(), &m, work.Off(iwrk1-1), toPtr((*lwork)-iwrk1+1), iwork, &ierr)

			}
		}
	}

	//     Undo balancing on VL and VR and normalization
	//     (Workspace: none needed)
	if ilvl {
		Zggbak(balanc, 'L', n, ilo, ihi, lscale, rscale, n, vl, ldvl, &ierr)

		for jc = 1; jc <= (*n); jc++ {
			temp = zero
			for jr = 1; jr <= (*n); jr++ {
				temp = maxf64(temp, abs1(vl.Get(jr-1, jc-1)))
			}
			if temp < smlnum {
				goto label50
			}
			temp = one / temp
			for jr = 1; jr <= (*n); jr++ {
				vl.Set(jr-1, jc-1, vl.Get(jr-1, jc-1)*toCmplx(temp))
			}
		label50:
		}
	}

	if ilvr {
		Zggbak(balanc, 'R', n, ilo, ihi, lscale, rscale, n, vr, ldvr, &ierr)
		for jc = 1; jc <= (*n); jc++ {
			temp = zero
			for jr = 1; jr <= (*n); jr++ {
				temp = maxf64(temp, abs1(vr.Get(jr-1, jc-1)))
			}
			if temp < smlnum {
				goto label80
			}
			temp = one / temp
			for jr = 1; jr <= (*n); jr++ {
				vr.Set(jr-1, jc-1, vr.Get(jr-1, jc-1)*toCmplx(temp))
			}
		label80:
		}
	}

	//     Undo scaling if necessary

label90:
	;

	if ilascl {
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrmto, &anrm, n, func() *int { y := 1; return &y }(), alpha.CMatrix(*n, opts), n, &ierr)
	}

	if ilbscl {
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrmto, &bnrm, n, func() *int { y := 1; return &y }(), beta.CMatrix(*n, opts), n, &ierr)
	}

	work.SetRe(0, float64(maxwrk))
}
