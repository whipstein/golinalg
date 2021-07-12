package golapack

import (
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dggevx computes for a pair of N-by-N real nonsymmetric matrices (A,B)
// the generalized eigenvalues, and optionally, the left and/or right
// generalized eigenvectors.
//
// Optionally also, it computes a balancing transformation to improve
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
//
//                  A * v(j) = lambda(j) * B * v(j) .
//
// The left eigenvector u(j) corresponding to the eigenvalue lambda(j)
// of (A,B) satisfies
//
//                  u(j)**H * A  = lambda(j) * u(j)**H * B.
//
// where u(j)**H is the conjugate-transpose of u(j).
func Dggevx(balanc, jobvl, jobvr, sense byte, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, alphar, alphai, beta *mat.Vector, vl *mat.Matrix, ldvl *int, vr *mat.Matrix, ldvr, ilo, ihi *int, lscale, rscale *mat.Vector, abnrm, bbnrm *float64, rconde, rcondv, work *mat.Vector, lwork *int, iwork *[]int, bwork *[]bool, info *int) {
	var ilascl, ilbscl, ilv, ilvl, ilvr, lquery, noscl, pair, wantsb, wantse, wantsn, wantsv bool
	var chtemp byte
	var anrm, anrmto, bignum, bnrm, bnrmto, eps, one, smlnum, temp, zero float64
	var i, icols, ierr, ijobvl, ijobvr, in, irows, itau, iwrk, iwrk1, j, jc, jr, m, maxwrk, minwrk, mm int

	ldumma := make([]bool, 1)

	zero = 0.0
	one = 1.0

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
	if !(balanc == 'N' || balanc == 'S' || balanc == 'P' || balanc == 'B') {
		(*info) = -1
	} else if ijobvl <= 0 {
		(*info) = -2
	} else if ijobvr <= 0 {
		(*info) = -3
	} else if !(wantsn || wantse || wantsb || wantsv) {
		(*info) = -4
	} else if (*n) < 0 {
		(*info) = -5
	} else if (*lda) < max(1, *n) {
		(*info) = -7
	} else if (*ldb) < max(1, *n) {
		(*info) = -9
	} else if (*ldvl) < 1 || (ilvl && (*ldvl) < (*n)) {
		(*info) = -14
	} else if (*ldvr) < 1 || (ilvr && (*ldvr) < (*n)) {
		(*info) = -16
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
			if noscl && !ilv {
				minwrk = 2 * (*n)
			} else {
				minwrk = 6 * (*n)
			}
			if wantse || wantsb {
				minwrk = 10 * (*n)
			}
			if wantsv || wantsb {
				minwrk = max(minwrk, 2*(*n)*((*n)+4)+16)
			}
			maxwrk = minwrk
			maxwrk = max(maxwrk, (*n)+(*n)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DGEQRF"), []byte{' '}, n, func() *int { y := 1; return &y }(), n, func() *int { y := 0; return &y }()))
			maxwrk = max(maxwrk, (*n)+(*n)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMQR"), []byte{' '}, n, func() *int { y := 1; return &y }(), n, func() *int { y := 0; return &y }()))
			if ilvl {
				maxwrk = max(maxwrk, (*n)+(*n)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DORGQR"), []byte{' '}, n, func() *int { y := 1; return &y }(), n, func() *int { y := 0; return &y }()))
			}
		}
		work.Set(0, float64(maxwrk))

		if (*lwork) < minwrk && !lquery {
			(*info) = -26
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGGEVX"), -(*info))
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

	//     Scale A if math.Max element outside range [SMLNUM,BIGNUM]
	anrm = Dlange('M', n, n, a, lda, work)
	ilascl = false
	if anrm > zero && anrm < smlnum {
		anrmto = smlnum
		ilascl = true
	} else if anrm > bignum {
		anrmto = bignum
		ilascl = true
	}
	if ilascl {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &anrmto, n, n, a, lda, &ierr)
	}

	//     Scale B if math.Max element outside range [SMLNUM,BIGNUM]
	bnrm = Dlange('M', n, n, b, ldb, work)
	ilbscl = false
	if bnrm > zero && bnrm < smlnum {
		bnrmto = smlnum
		ilbscl = true
	} else if bnrm > bignum {
		bnrmto = bignum
		ilbscl = true
	}
	if ilbscl {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrm, &bnrmto, n, n, b, ldb, &ierr)
	}

	//     Permute and/or balance the matrix pair (A,B)
	//     (Workspace: need 6*N if BALANC = 'S' or 'B', 1 otherwise)
	Dggbal(balanc, n, a, lda, b, ldb, ilo, ihi, lscale, rscale, work, &ierr)

	//     Compute ABNRM and BBNRM
	(*abnrm) = Dlange('1', n, n, a, lda, work.Off(0))
	if ilascl {
		work.Set(0, (*abnrm))
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrmto, &anrm, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), work.Matrix(1, opts), func() *int { y := 1; return &y }(), &ierr)
		(*abnrm) = work.Get(0)
	}

	(*bbnrm) = Dlange('1', n, n, b, ldb, work.Off(0))
	if ilbscl {
		work.Set(0, (*bbnrm))
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrmto, &bnrm, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), work.Matrix(1, opts), func() *int { y := 1; return &y }(), &ierr)
		(*bbnrm) = work.Get(0)
	}

	//     Reduce B to triangular form (QR decomposition of B)
	//     (Workspace: need N, prefer N*NB )
	irows = (*ihi) + 1 - (*ilo)
	if ilv || !wantsn {
		icols = (*n) + 1 - (*ilo)
	} else {
		icols = irows
	}
	itau = 1
	iwrk = itau + irows
	Dgeqrf(&irows, &icols, b.Off((*ilo)-1, (*ilo)-1), ldb, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)

	//     Apply the orthogonal transformation to A
	//     (Workspace: need N, prefer N*NB)
	Dormqr('L', 'T', &irows, &icols, &irows, b.Off((*ilo)-1, (*ilo)-1), ldb, work.Off(itau-1), a.Off((*ilo)-1, (*ilo)-1), lda, work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)

	//     Initialize VL and/or VR
	//     (Workspace: need N, prefer N*NB)
	if ilvl {
		Dlaset('F', n, n, &zero, &one, vl, ldvl)
		if irows > 1 {
			Dlacpy('L', toPtr(irows-1), toPtr(irows-1), b.Off((*ilo), (*ilo)-1), ldb, vl.Off((*ilo), (*ilo)-1), ldvl)
		}
		Dorgqr(&irows, &irows, &irows, vl.Off((*ilo)-1, (*ilo)-1), ldvl, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)
	}

	if ilvr {
		Dlaset('F', n, n, &zero, &one, vr, ldvr)
	}

	//     Reduce to generalized Hessenberg form
	//     (Workspace: none needed)
	if ilv || !wantsn {
		//        Eigenvectors requested -- work on whole matrix.
		Dgghrd(jobvl, jobvr, n, ilo, ihi, a, lda, b, ldb, vl, ldvl, vr, ldvr, &ierr)
	} else {
		Dgghrd('N', 'N', &irows, func() *int { y := 1; return &y }(), &irows, a.Off((*ilo)-1, (*ilo)-1), lda, b.Off((*ilo)-1, (*ilo)-1), ldb, vl, ldvl, vr, ldvr, &ierr)
	}

	//     Perform QZ algorithm (Compute eigenvalues, and optionally, the
	//     Schur forms and Schur vectors)
	//     (Workspace: need N)
	if ilv || !wantsn {
		chtemp = 'S'
	} else {
		chtemp = 'E'
	}

	Dhgeqz(chtemp, jobvl, jobvr, n, ilo, ihi, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr, work, lwork, &ierr)
	if ierr != 0 {
		if ierr > 0 && ierr <= (*n) {
			(*info) = ierr
		} else if ierr > (*n) && ierr <= 2*(*n) {
			(*info) = ierr - (*n)
		} else {
			(*info) = (*n) + 1
		}
		goto label130
	}

	//     Compute Eigenvectors and estimate condition numbers if desired
	//     (Workspace: DTGEVC: need 6*N
	//                 DTGSNA: need 2*N*(N+2)+16 if SENSE = 'V' or 'B',
	//                         need N otherwise )
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

			Dtgevc(chtemp, 'B', ldumma, n, a, lda, b, ldb, vl, ldvl, vr, ldvr, n, &in, work, &ierr)
			if ierr != 0 {
				(*info) = (*n) + 2
				goto label130
			}
		}

		if !wantsn {
			//           compute eigenvectors (DTGEVC) and estimate condition
			//           numbers (DTGSNA). Note that the definition of the condition
			//           number is not invariant under transformation (u,v) to
			//           (Q*u, Z*v), where (u,v) are eigenvectors of the generalized
			//           Schur form (S,T), Q and Z are orthogonal matrices. In order
			//           to avoid using extra 2*N*N workspace, we have to recalculate
			//           eigenvectors and estimate one condition numbers at a time.
			pair = false
			for i = 1; i <= (*n); i++ {

				if pair {
					pair = false
					goto label20
				}
				mm = 1
				if i < (*n) {
					if a.Get(i, i-1) != zero {
						pair = true
						mm = 2
					}
				}

				for j = 1; j <= (*n); j++ {
					(*bwork)[j-1] = false
				}
				if mm == 1 {
					(*bwork)[i-1] = true
				} else if mm == 2 {
					(*bwork)[i-1] = true
					(*bwork)[i] = true
				}

				iwrk = mm*(*n) + 1
				iwrk1 = iwrk + mm*(*n)

				//              Compute a pair of left and right eigenvectors.
				//              (compute workspace: need up to 4*N + 6*N)
				if wantse || wantsb {
					Dtgevc('B', 'S', *bwork, n, a, lda, b, ldb, work.Matrix(*n, opts), n, work.MatrixOff(iwrk-1, *n, opts), n, &mm, &m, work.Off(iwrk1-1), &ierr)
					if ierr != 0 {
						(*info) = (*n) + 2
						goto label130
					}
				}

				Dtgsna(sense, 'S', *bwork, n, a, lda, b, ldb, work.Matrix(*n, opts), n, work.MatrixOff(iwrk-1, *n, opts), n, rconde.Off(i-1), rcondv.Off(i-1), &mm, &m, work.Off(iwrk1-1), toPtr((*lwork)-iwrk1+1), iwork, &ierr)

			label20:
			}
		}
	}

	//     Undo balancing on VL and VR and normalization
	//     (Workspace: none needed)
	if ilvl {
		Dggbak(balanc, 'L', n, ilo, ihi, lscale, rscale, n, vl, ldvl, &ierr)

		for jc = 1; jc <= (*n); jc++ {
			if alphai.Get(jc-1) < zero {
				goto label70
			}
			temp = zero
			if alphai.Get(jc-1) == zero {
				for jr = 1; jr <= (*n); jr++ {
					temp = math.Max(temp, math.Abs(vl.Get(jr-1, jc-1)))
				}
			} else {
				for jr = 1; jr <= (*n); jr++ {
					temp = math.Max(temp, math.Abs(vl.Get(jr-1, jc-1))+math.Abs(vl.Get(jr-1, jc)))
				}
			}
			if temp < smlnum {
				goto label70
			}
			temp = one / temp
			if alphai.Get(jc-1) == zero {
				for jr = 1; jr <= (*n); jr++ {
					vl.Set(jr-1, jc-1, vl.Get(jr-1, jc-1)*temp)
				}
			} else {
				for jr = 1; jr <= (*n); jr++ {
					vl.Set(jr-1, jc-1, vl.Get(jr-1, jc-1)*temp)
					vl.Set(jr-1, jc, vl.Get(jr-1, jc)*temp)
				}
			}
		label70:
		}
	}
	if ilvr {
		Dggbak(balanc, 'R', n, ilo, ihi, lscale, rscale, n, vr, ldvr, &ierr)
		for jc = 1; jc <= (*n); jc++ {
			if alphai.Get(jc-1) < zero {
				goto label120
			}
			temp = zero
			if alphai.Get(jc-1) == zero {
				for jr = 1; jr <= (*n); jr++ {
					temp = math.Max(temp, math.Abs(vr.Get(jr-1, jc-1)))
				}
			} else {
				for jr = 1; jr <= (*n); jr++ {
					temp = math.Max(temp, math.Abs(vr.Get(jr-1, jc-1))+math.Abs(vr.Get(jr-1, jc)))
				}
			}
			if temp < smlnum {
				goto label120
			}
			temp = one / temp
			if alphai.Get(jc-1) == zero {
				for jr = 1; jr <= (*n); jr++ {
					vr.Set(jr-1, jc-1, vr.Get(jr-1, jc-1)*temp)
				}
			} else {
				for jr = 1; jr <= (*n); jr++ {
					vr.Set(jr-1, jc-1, vr.Get(jr-1, jc-1)*temp)
					vr.Set(jr-1, jc, vr.Get(jr-1, jc)*temp)
				}
			}
		label120:
		}
	}

	//     Undo scaling if necessary
label130:
	;

	if ilascl {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrmto, &anrm, n, func() *int { y := 1; return &y }(), alphar.Matrix(*n, opts), n, &ierr)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrmto, &anrm, n, func() *int { y := 1; return &y }(), alphai.Matrix(*n, opts), n, &ierr)
	}

	if ilbscl {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrmto, &bnrm, n, func() *int { y := 1; return &y }(), beta.Matrix(*n, opts), n, &ierr)
	}

	work.Set(0, float64(maxwrk))
}
