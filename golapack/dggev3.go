package golapack

import (
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dggev3 computes for a pair of N-by-N real nonsymmetric matrices (A,B)
// the generalized eigenvalues, and optionally, the left and/or right
// generalized eigenvectors.
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
//                  A * v(j) = lambda(j) * B * v(j).
//
// The left eigenvector u(j) corresponding to the eigenvalue lambda(j)
// of (A,B) satisfies
//
//                  u(j)**H * A  = lambda(j) * u(j)**H * B .
//
// where u(j)**H is the conjugate-transpose of u(j).
func Dggev3(jobvl, jobvr byte, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, alphar, alphai, beta *mat.Vector, vl *mat.Matrix, ldvl *int, vr *mat.Matrix, ldvr *int, work *mat.Vector, lwork, info *int) {
	var ilascl, ilbscl, ilv, ilvl, ilvr, lquery bool
	var chtemp byte
	var anrm, anrmto, bignum, bnrm, bnrmto, eps, one, smlnum, temp, zero float64
	var icols, ierr, ihi, ijobvl, ijobvr, ileft, ilo, in, iright, irows, itau, iwrk, jc, jr, lwkopt int

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

	//     Test the input arguments
	(*info) = 0
	lquery = ((*lwork) == -1)
	if ijobvl <= 0 {
		(*info) = -1
	} else if ijobvr <= 0 {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*lda) < max(1, *n) {
		(*info) = -5
	} else if (*ldb) < max(1, *n) {
		(*info) = -7
	} else if (*ldvl) < 1 || (ilvl && (*ldvl) < (*n)) {
		(*info) = -12
	} else if (*ldvr) < 1 || (ilvr && (*ldvr) < (*n)) {
		(*info) = -14
	} else if (*lwork) < max(1, 8*(*n)) && !lquery {
		(*info) = -16
	}

	//     Compute workspace
	if (*info) == 0 {
		Dgeqrf(n, n, b, ldb, work, work, toPtr(-1), &ierr)
		lwkopt = max(1, 8*(*n), 3*(*n)+int(work.Get(0)))
		Dormqr('L', 'T', n, n, n, b, ldb, work, a, lda, work, toPtr(-1), &ierr)
		lwkopt = max(lwkopt, 3*(*n)+int(work.Get(0)))
		if ilvl {
			Dorgqr(n, n, n, vl, ldvl, work, work, toPtr(-1), &ierr)
			lwkopt = max(lwkopt, 3*(*n)+int(work.Get(0)))
		}
		if ilv {
			Dgghd3(jobvl, jobvr, n, func() *int { y := 1; return &y }(), n, a, lda, b, ldb, vl, ldvl, vr, ldvr, work, toPtr(-1), &ierr)
			lwkopt = max(lwkopt, 3*(*n)+int(work.Get(0)))
			Dhgeqz('S', jobvl, jobvr, n, func() *int { y := 1; return &y }(), n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr, work, toPtr(-1), &ierr)
			lwkopt = max(lwkopt, 2*(*n)+int(work.Get(0)))
		} else {
			Dgghd3('N', 'N', n, func() *int { y := 1; return &y }(), n, a, lda, b, ldb, vl, ldvl, vr, ldvr, work, toPtr(-1), &ierr)
			lwkopt = max(lwkopt, 3*(*n)+int(work.Get(0)))
			Dhgeqz('E', jobvl, jobvr, n, func() *int { y := 1; return &y }(), n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr, work, toPtr(-1), &ierr)
			lwkopt = max(lwkopt, 2*(*n)+int(work.Get(0)))
		}
		work.Set(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGGEV3 "), -(*info))
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

	//     Scale A if max element outside range [SMLNUM,BIGNUM]
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

	//     Scale B if max element outside range [SMLNUM,BIGNUM]
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

	//     Permute the matrices A, B to isolate eigenvalues if possible
	ileft = 1
	iright = (*n) + 1
	iwrk = iright + (*n)
	Dggbal('P', n, a, lda, b, ldb, &ilo, &ihi, work.Off(ileft-1), work.Off(iright-1), work.Off(iwrk-1), &ierr)

	//     Reduce B to triangular form (QR decomposition of B)
	irows = ihi + 1 - ilo
	if ilv {
		icols = (*n) + 1 - ilo
	} else {
		icols = irows
	}
	itau = iwrk
	iwrk = itau + irows
	Dgeqrf(&irows, &icols, b.Off(ilo-1, ilo-1), ldb, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)

	//     Apply the orthogonal transformation to matrix A
	Dormqr('L', 'T', &irows, &icols, &irows, b.Off(ilo-1, ilo-1), ldb, work.Off(itau-1), a.Off(ilo-1, ilo-1), lda, work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)

	//     Initialize VL
	if ilvl {
		Dlaset('F', n, n, &zero, &one, vl, ldvl)
		if irows > 1 {
			Dlacpy('L', toPtr(irows-1), toPtr(irows-1), b.Off(ilo, ilo-1), ldb, vl.Off(ilo, ilo-1), ldvl)
		}
		Dorgqr(&irows, &irows, &irows, vl.Off(ilo-1, ilo-1), ldvl, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)
	}

	//     Initialize VR
	if ilvr {
		Dlaset('F', n, n, &zero, &one, vr, ldvr)
	}

	//     Reduce to generalized Hessenberg form
	if ilv {
		//        Eigenvectors requested -- work on whole matrix.
		Dgghd3(jobvl, jobvr, n, &ilo, &ihi, a, lda, b, ldb, vl, ldvl, vr, ldvr, work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)
	} else {
		Dgghd3('N', 'N', &irows, func() *int { y := 1; return &y }(), &irows, a.Off(ilo-1, ilo-1), lda, b.Off(ilo-1, ilo-1), ldb, vl, ldvl, vr, ldvr, work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)
	}

	//     Perform QZ algorithm (Compute eigenvalues, and optionally, the
	//     Schur forms and Schur vectors)
	iwrk = itau
	if ilv {
		chtemp = 'S'
	} else {
		chtemp = 'E'
	}
	Dhgeqz(chtemp, jobvl, jobvr, n, &ilo, &ihi, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr, work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)
	if ierr != 0 {
		if ierr > 0 && ierr <= (*n) {
			(*info) = ierr
		} else if ierr > (*n) && ierr <= 2*(*n) {
			(*info) = ierr - (*n)
		} else {
			(*info) = (*n) + 1
		}
		goto label110
	}

	//     Compute Eigenvectors
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
		Dtgevc(chtemp, 'B', ldumma, n, a, lda, b, ldb, vl, ldvl, vr, ldvr, n, &in, work.Off(iwrk-1), &ierr)
		if ierr != 0 {
			(*info) = (*n) + 2
			goto label110
		}

		//        Undo balancing on VL and VR and normalization
		if ilvl {
			Dggbak('P', 'L', n, &ilo, &ihi, work.Off(ileft-1), work.Off(iright-1), n, vl, ldvl, &ierr)
			for jc = 1; jc <= (*n); jc++ {
				if alphai.Get(jc-1) < zero {
					goto label50
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
					goto label50
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
			label50:
			}
		}
		if ilvr {
			Dggbak('P', 'R', n, &ilo, &ihi, work.Off(ileft-1), work.Off(iright-1), n, vr, ldvr, &ierr)
			for jc = 1; jc <= (*n); jc++ {
				if alphai.Get(jc-1) < zero {
					goto label100
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
					goto label100
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
			label100:
			}
		}

		//        End of eigenvector calculation
	}

	//     Undo scaling if necessary
label110:
	;

	if ilascl {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrmto, &anrm, n, func() *int { y := 1; return &y }(), alphar.Matrix(*n, opts), n, &ierr)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrmto, &anrm, n, func() *int { y := 1; return &y }(), alphai.Matrix(*n, opts), n, &ierr)
	}

	if ilbscl {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrmto, &bnrm, n, func() *int { y := 1; return &y }(), beta.Matrix(*n, opts), n, &ierr)
	}

	work.Set(0, float64(lwkopt))
}
