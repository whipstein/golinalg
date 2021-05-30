package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Zggesx computes for a pair of N-by-N complex nonsymmetric matrices
// (A,B), the generalized eigenvalues, the complex Schur form (S,T),
// and, optionally, the left and/or right matrices of Schur vectors (VSL
// and VSR).  This gives the generalized Schur factorization
//
//      (A,B) = ( (VSL) S (VSR)**H, (VSL) T (VSR)**H )
//
// where (VSR)**H is the conjugate-transpose of VSR.
//
// Optionally, it also orders the eigenvalues so that a selected cluster
// of eigenvalues appears in the leading diagonal blocks of the upper
// triangular matrix S and the upper triangular matrix T; computes
// a reciprocal condition number for the average of the selected
// eigenvalues (RCONDE); and computes a reciprocal condition number for
// the right and left deflating subspaces corresponding to the selected
// eigenvalues (RCONDV). The leading columns of VSL and VSR then form
// an orthonormal basis for the corresponding left and right eigenspaces
// (deflating subspaces).
//
// A generalized eigenvalue for a pair of matrices (A,B) is a scalar w
// or a ratio alpha/beta = w, such that  A - w*B is singular.  It is
// usually represented as the pair (alpha,beta), as there is a
// reasonable interpretation for beta=0 or for both being zero.
//
// A pair of matrices (S,T) is in generalized complex Schur form if T is
// upper triangular with non-negative diagonal and S is upper
// triangular.
func Zggesx(jobvsl, jobvsr, sort byte, selctg func(complex128, complex128) bool, sense byte, n *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, sdim *int, alpha, beta *mat.CVector, vsl *mat.CMatrix, ldvsl *int, vsr *mat.CMatrix, ldvsr *int, rconde, rcondv *mat.Vector, work *mat.CVector, lwork *int, rwork *mat.Vector, iwork *[]int, liwork *int, bwork *[]bool, info *int) {
	var cursl, ilascl, ilbscl, ilvsl, ilvsr, lastsl, lquery, wantsb, wantse, wantsn, wantst, wantsv bool
	var cone, czero complex128
	var anrm, anrmto, bignum, bnrm, bnrmto, eps, one, pl, pr, smlnum, zero float64
	var i, icols, ierr, ihi, ijob, ijobvl, ijobvr, ileft, ilo, iright, irows, irwrk, itau, iwrk, liwmin, lwrk, maxwrk, minwrk int
	dif := vf(2)

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Decode the input arguments
	if jobvsl == 'N' {
		ijobvl = 1
		ilvsl = false
	} else if jobvsl == 'V' {
		ijobvl = 2
		ilvsl = true
	} else {
		ijobvl = -1
		ilvsl = false
	}

	if jobvsr == 'N' {
		ijobvr = 1
		ilvsr = false
	} else if jobvsr == 'V' {
		ijobvr = 2
		ilvsr = true
	} else {
		ijobvr = -1
		ilvsr = false
	}

	wantst = sort == 'S'
	wantsn = sense == 'N'
	wantse = sense == 'E'
	wantsv = sense == 'V'
	wantsb = sense == 'B'
	lquery = ((*lwork) == -1 || (*liwork) == -1)
	if wantsn {
		ijob = 0
	} else if wantse {
		ijob = 1
	} else if wantsv {
		ijob = 2
	} else if wantsb {
		ijob = 4
	}

	//     Test the input arguments
	(*info) = 0
	if ijobvl <= 0 {
		(*info) = -1
	} else if ijobvr <= 0 {
		(*info) = -2
	} else if (!wantst) && (sort != 'N') {
		(*info) = -3
	} else if !(wantsn || wantse || wantsv || wantsb) || (!wantst && !wantsn) {
		(*info) = -5
	} else if (*n) < 0 {
		(*info) = -6
	} else if (*lda) < maxint(1, *n) {
		(*info) = -8
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -10
	} else if (*ldvsl) < 1 || (ilvsl && (*ldvsl) < (*n)) {
		(*info) = -15
	} else if (*ldvsr) < 1 || (ilvsr && (*ldvsr) < (*n)) {
		(*info) = -17
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       NB refers to the optimal block size for the immediately
	//       following subroutine, as returned by ILAENV.)
	if (*info) == 0 {
		if (*n) > 0 {
			minwrk = 2 * (*n)
			maxwrk = (*n) * (1 + Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEQRF"), []byte{' '}, n, func() *int { y := 1; return &y }(), n, func() *int { y := 0; return &y }()))
			maxwrk = maxint(maxwrk, (*n)*(1+Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMQR"), []byte{' '}, n, func() *int { y := 1; return &y }(), n, toPtr(-1))))
			if ilvsl {
				maxwrk = maxint(maxwrk, (*n)*(1+Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNGQR"), []byte{' '}, n, func() *int { y := 1; return &y }(), n, toPtr(-1))))
			}
			lwrk = maxwrk
			if ijob >= 1 {
				lwrk = maxint(lwrk, (*n)*(*n)/2)
			}
		} else {
			minwrk = 1
			maxwrk = 1
			lwrk = 1
		}
		work.SetRe(0, float64(lwrk))
		if wantsn || (*n) == 0 {
			liwmin = 1
		} else {
			liwmin = (*n) + 2
		}
		(*iwork)[0] = liwmin

		if (*lwork) < minwrk && !lquery {
			(*info) = -21
		} else if (*liwork) < liwmin && !lquery {
			(*info) = -24
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGGESX"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		(*sdim) = 0
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

	//     Permute the matrix to make it more nearly triangular
	//     (Real Workspace: need 6*N)
	ileft = 1
	iright = (*n) + 1
	irwrk = iright + (*n)
	Zggbal('P', n, a, lda, b, ldb, &ilo, &ihi, rwork.Off(ileft-1), rwork.Off(iright-1), rwork.Off(irwrk-1), &ierr)

	//     Reduce B to triangular form (QR decomposition of B)
	//     (Complex Workspace: need N, prefer N*NB)
	irows = ihi + 1 - ilo
	icols = (*n) + 1 - ilo
	itau = 1
	iwrk = itau + irows
	Zgeqrf(&irows, &icols, b.Off(ilo-1, ilo-1), ldb, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)

	//     Apply the unitary transformation to matrix A
	//     (Complex Workspace: need N, prefer N*NB)
	Zunmqr('L', 'C', &irows, &icols, &irows, b.Off(ilo-1, ilo-1), ldb, work.Off(itau-1), a.Off(ilo-1, ilo-1), lda, work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)

	//     Initialize VSL
	//     (Complex Workspace: need N, prefer N*NB)
	if ilvsl {
		Zlaset('F', n, n, &czero, &cone, vsl, ldvsl)
		if irows > 1 {
			Zlacpy('L', toPtr(irows-1), toPtr(irows-1), b.Off(ilo+1-1, ilo-1), ldb, vsl.Off(ilo+1-1, ilo-1), ldvsl)
		}
		Zungqr(&irows, &irows, &irows, vsl.Off(ilo-1, ilo-1), ldvsl, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)
	}

	//     Initialize VSR
	if ilvsr {
		Zlaset('F', n, n, &czero, &cone, vsr, ldvsr)
	}

	//     Reduce to generalized Hessenberg form
	//     (Workspace: none needed)
	Zgghrd(jobvsl, jobvsr, n, &ilo, &ihi, a, lda, b, ldb, vsl, ldvsl, vsr, ldvsr, &ierr)

	(*sdim) = 0

	//     Perform QZ algorithm, computing Schur vectors if desired
	//     (Complex Workspace: need N)
	//     (Real Workspace:    need N)
	iwrk = itau
	Zhgeqz('S', jobvsl, jobvsr, n, &ilo, &ihi, a, lda, b, ldb, alpha, beta, vsl, ldvsl, vsr, ldvsr, work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), rwork.Off(irwrk-1), &ierr)
	if ierr != 0 {
		if ierr > 0 && ierr <= (*n) {
			(*info) = ierr
		} else if ierr > (*n) && ierr <= 2*(*n) {
			(*info) = ierr - (*n)
		} else {
			(*info) = (*n) + 1
		}
		goto label40
	}

	//     Sort eigenvalues ALPHA/BETA and compute the reciprocal of
	//     condition number(s)
	if wantst {
		//        Undo scaling on eigenvalues before SELCTGing
		if ilascl {
			Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrmto, &anrm, n, func() *int { y := 1; return &y }(), alpha.CMatrix(*n, opts), n, &ierr)
		}
		if ilbscl {
			Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrmto, &bnrm, n, func() *int { y := 1; return &y }(), beta.CMatrix(*n, opts), n, &ierr)
		}

		//        Select eigenvalues
		for i = 1; i <= (*n); i++ {
			(*bwork)[i-1] = selctg(alpha.Get(i-1), beta.Get(i-1))
		}

		//        Reorder eigenvalues, transform Generalized Schur vectors, and
		//        compute reciprocal condition numbers
		//        (Complex Workspace: If IJOB >= 1, need MAX(1, 2*SDIM*(N-SDIM))
		//                            otherwise, need 1 )
		Ztgsen(&ijob, ilvsl, ilvsr, *bwork, n, a, lda, b, ldb, alpha, beta, vsl, ldvsl, vsr, ldvsr, sdim, &pl, &pr, dif, work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), iwork, liwork, &ierr)

		if ijob >= 1 {
			maxwrk = maxint(maxwrk, 2*(*sdim)*((*n)-(*sdim)))
		}
		if ierr == -21 {
			//            not enough complex workspace
			(*info) = -21
		} else {
			if ijob == 1 || ijob == 4 {
				rconde.Set(0, pl)
				rconde.Set(1, pr)
			}
			if ijob == 2 || ijob == 4 {
				rcondv.Set(0, dif.Get(0))
				rcondv.Set(1, dif.Get(1))
			}
			if ierr == 1 {
				(*info) = (*n) + 3
			}
		}

	}

	//     Apply permutation to VSL and VSR
	//     (Workspace: none needed)
	if ilvsl {
		Zggbak('P', 'L', n, &ilo, &ihi, rwork.Off(ileft-1), rwork.Off(iright-1), n, vsl, ldvsl, &ierr)
	}

	if ilvsr {
		Zggbak('P', 'R', n, &ilo, &ihi, rwork.Off(ileft-1), rwork.Off(iright-1), n, vsr, ldvsr, &ierr)
	}

	//     Undo scaling
	if ilascl {
		Zlascl('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrmto, &anrm, n, n, a, lda, &ierr)
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrmto, &anrm, n, func() *int { y := 1; return &y }(), alpha.CMatrix(*n, opts), n, &ierr)
	}

	if ilbscl {
		Zlascl('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrmto, &bnrm, n, n, b, ldb, &ierr)
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrmto, &bnrm, n, func() *int { y := 1; return &y }(), beta.CMatrix(*n, opts), n, &ierr)
	}

	if wantst {
		//        Check if reordering is correct
		lastsl = true
		(*sdim) = 0
		for i = 1; i <= (*n); i++ {
			cursl = selctg(alpha.Get(i-1), beta.Get(i-1))
			if cursl {
				(*sdim) = (*sdim) + 1
			}
			if cursl && !lastsl {
				(*info) = (*n) + 2
			}
			lastsl = cursl
		}

	}

label40:
	;

	work.SetRe(0, float64(maxwrk))
	(*iwork)[0] = liwmin
}
