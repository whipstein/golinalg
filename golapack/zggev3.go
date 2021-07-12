package golapack

import (
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zggev3 computes for a pair of N-by-N complex nonsymmetric matrices
// (A,B), the generalized eigenvalues, and optionally, the left and/or
// right generalized eigenvectors.
//
// A generalized eigenvalue for a pair of matrices (A,B) is a scalar
// lambda or a ratio alpha/beta = lambda, such that A - lambda*B is
// singular. It is usually represented as the pair (alpha,beta), as
// there is a reasonable interpretation for beta=0, and even for both
// being zero.
//
// The right generalized eigenvector v(j) corresponding to the
// generalized eigenvalue lambda(j) of (A,B) satisfies
//
//              A * v(j) = lambda(j) * B * v(j).
//
// The left generalized eigenvector u(j) corresponding to the
// generalized eigenvalues lambda(j) of (A,B) satisfies
//
//              u(j)**H * A = lambda(j) * u(j)**H * B
//
// where u(j)**H is the conjugate-transpose of u(j).
func Zggev3(jobvl, jobvr byte, n *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, alpha, beta *mat.CVector, vl *mat.CMatrix, ldvl *int, vr *mat.CMatrix, ldvr *int, work *mat.CVector, lwork *int, rwork *mat.Vector, info *int) {
	var ilascl, ilbscl, ilv, ilvl, ilvr, lquery bool
	var chtemp byte
	var cone, czero complex128
	var anrm, anrmto, bignum, bnrm, bnrmto, eps, one, smlnum, temp, zero float64
	var icols, ierr, ihi, ijobvl, ijobvr, ileft, ilo, in, iright, irows, irwrk, itau, iwrk, jc, jr, lwkopt int
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
		(*info) = -11
	} else if (*ldvr) < 1 || (ilvr && (*ldvr) < (*n)) {
		(*info) = -13
	} else if (*lwork) < max(1, 2*(*n)) && !lquery {
		(*info) = -15
	}

	//     Compute workspace
	if (*info) == 0 {
		Zgeqrf(n, n, b, ldb, work, work, toPtr(-1), &ierr)
		lwkopt = max(1, (*n)+int(work.GetRe(0)))
		Zunmqr('L', 'C', n, n, n, b, ldb, work, a, lda, work, toPtr(-1), &ierr)
		lwkopt = max(lwkopt, (*n)+int(work.GetRe(0)))
		if ilvl {
			Zungqr(n, n, n, vl, ldvl, work, work, toPtr(-1), &ierr)
			lwkopt = max(lwkopt, (*n)+int(work.GetRe(0)))
		}
		if ilv {
			Zgghd3(jobvl, jobvr, n, func() *int { y := 1; return &y }(), n, a, lda, b, ldb, vl, ldvl, vr, ldvr, work, toPtr(-1), &ierr)
			lwkopt = max(lwkopt, (*n)+int(work.GetRe(0)))
			Zhgeqz('S', jobvl, jobvr, n, func() *int { y := 1; return &y }(), n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr, work, toPtr(-1), rwork, &ierr)
			lwkopt = max(lwkopt, (*n)+int(work.GetRe(0)))
		} else {
			Zgghd3(jobvl, jobvr, n, func() *int { y := 1; return &y }(), n, a, lda, b, ldb, vl, ldvl, vr, ldvr, work, toPtr(-1), &ierr)
			lwkopt = max(lwkopt, (*n)+int(work.GetRe(0)))
			Zhgeqz('E', jobvl, jobvr, n, func() *int { y := 1; return &y }(), n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr, work, toPtr(-1), rwork, &ierr)
			lwkopt = max(lwkopt, (*n)+int(work.GetRe(0)))
		}
		work.SetRe(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGGEV3"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Get machine constants
	eps = Dlamch(Epsilon) * Dlamch(Base)
	smlnum = Dlamch(SafeMinimum)
	bignum = one / smlnum
	Dlabad(&smlnum, &bignum)
	smlnum = math.Sqrt(smlnum) / eps
	bignum = one / smlnum

	//     Scale A if max element outside range [SMLNUM,BIGNUM]
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

	//     Scale B if max element outside range [SMLNUM,BIGNUM]
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

	//     Permute the matrices A, B to isolate eigenvalues if possible
	ileft = 1
	iright = (*n) + 1
	irwrk = iright + (*n)
	Zggbal('P', n, a, lda, b, ldb, &ilo, &ihi, rwork.Off(ileft-1), rwork.Off(iright-1), rwork.Off(irwrk-1), &ierr)

	//     Reduce B to triangular form (QR decomposition of B)
	irows = ihi + 1 - ilo
	if ilv {
		icols = (*n) + 1 - ilo
	} else {
		icols = irows
	}
	itau = 1
	iwrk = itau + irows
	Zgeqrf(&irows, &icols, b.Off(ilo-1, ilo-1), ldb, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)

	//     Apply the orthogonal transformation to matrix A
	Zunmqr('L', 'C', &irows, &icols, &irows, b.Off(ilo-1, ilo-1), ldb, work.Off(itau-1), a.Off(ilo-1, ilo-1), lda, work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)

	//     Initialize VL
	if ilvl {
		Zlaset('F', n, n, &czero, &cone, vl, ldvl)
		if irows > 1 {
			Zlacpy('L', toPtr(irows-1), toPtr(irows-1), b.Off(ilo, ilo-1), ldb, vl.Off(ilo, ilo-1), ldvl)
		}
		Zungqr(&irows, &irows, &irows, vl.Off(ilo-1, ilo-1), ldvl, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)
	}

	//     Initialize VR
	if ilvr {
		Zlaset('F', n, n, &czero, &cone, vr, ldvr)
	}

	//     Reduce to generalized Hessenberg form
	if ilv {
		//        Eigenvectors requested -- work on whole matrix.
		Zgghd3(jobvl, jobvr, n, &ilo, &ihi, a, lda, b, ldb, vl, ldvl, vr, ldvr, work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)
	} else {
		Zgghd3('N', 'N', &irows, func() *int { y := 1; return &y }(), &irows, a.Off(ilo-1, ilo-1), lda, b.Off(ilo-1, ilo-1), ldb, vl, ldvl, vr, ldvr, work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)
	}

	//     Perform QZ algorithm (Compute eigenvalues, and optionally, the
	//     Schur form and Schur vectors)
	iwrk = itau
	if ilv {
		chtemp = 'S'
	} else {
		chtemp = 'E'
	}
	Zhgeqz(chtemp, jobvl, jobvr, n, &ilo, &ihi, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr, work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), rwork.Off(irwrk-1), &ierr)
	if ierr != 0 {
		if ierr > 0 && ierr <= (*n) {
			(*info) = ierr
		} else if ierr > (*n) && ierr <= 2*(*n) {
			(*info) = ierr - (*n)
		} else {
			(*info) = (*n) + 1
		}
		goto label70
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

		Ztgevc(chtemp, 'B', ldumma, n, a, lda, b, ldb, vl, ldvl, vr, ldvr, n, &in, work.Off(iwrk-1), rwork.Off(irwrk-1), &ierr)
		if ierr != 0 {
			(*info) = (*n) + 2
			goto label70
		}

		//        Undo balancing on VL and VR and normalization
		if ilvl {
			Zggbak('P', 'L', n, &ilo, &ihi, rwork.Off(ileft-1), rwork.Off(iright-1), n, vl, ldvl, &ierr)
			for jc = 1; jc <= (*n); jc++ {
				temp = zero
				for jr = 1; jr <= (*n); jr++ {
					temp = math.Max(temp, abs1(vl.Get(jr-1, jc-1)))
				}
				if temp < smlnum {
					goto label30
				}
				temp = one / temp
				for jr = 1; jr <= (*n); jr++ {
					vl.Set(jr-1, jc-1, vl.Get(jr-1, jc-1)*toCmplx(temp))
				}
			label30:
			}
		}
		if ilvr {
			Zggbak('P', 'R', n, &ilo, &ihi, rwork.Off(ileft-1), rwork.Off(iright-1), n, vr, ldvr, &ierr)
			for jc = 1; jc <= (*n); jc++ {
				temp = zero
				for jr = 1; jr <= (*n); jr++ {
					temp = math.Max(temp, abs1(vr.Get(jr-1, jc-1)))
				}
				if temp < smlnum {
					goto label60
				}
				temp = one / temp
				for jr = 1; jr <= (*n); jr++ {
					vr.Set(jr-1, jc-1, vr.Get(jr-1, jc-1)*toCmplx(temp))
				}
			label60:
			}
		}
	}

	//     Undo scaling if necessary
label70:
	;

	if ilascl {
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrmto, &anrm, n, func() *int { y := 1; return &y }(), alpha.CMatrix(*n, opts), n, &ierr)
	}

	if ilbscl {
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrmto, &bnrm, n, func() *int { y := 1; return &y }(), beta.CMatrix(*n, opts), n, &ierr)
	}

	work.SetRe(0, float64(lwkopt))
}
