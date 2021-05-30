package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Zgeevx computes for an N-by-N complex nonsymmetric matrix A, the
// eigenvalues and, optionally, the left and/or right eigenvectors.
//
// Optionally also, it computes a balancing transformation to improve
// the conditioning of the eigenvalues and eigenvectors (ILO, IHI,
// SCALE, and ABNRM), reciprocal condition numbers for the eigenvalues
// (RCONDE), and reciprocal condition numbers for the right
// eigenvectors (RCONDV).
//
// The right eigenvector v(j) of A satisfies
//                  A * v(j) = lambda(j) * v(j)
// where lambda(j) is its eigenvalue.
// The left eigenvector u(j) of A satisfies
//               u(j)**H * A = lambda(j) * u(j)**H
// where u(j)**H denotes the conjugate transpose of u(j).
//
// The computed eigenvectors are normalized to have Euclidean norm
// equal to 1 and largest component real.
//
// Balancing a matrix means permuting the rows and columns to make it
// more nearly upper triangular, and applying a diagonal similarity
// transformation D * A * D**(-1), where D is a diagonal matrix, to
// make its rows and columns closer in norm and the condition numbers
// of its eigenvalues and eigenvectors smaller.  The computed
// reciprocal condition numbers correspond to the balanced matrix.
// Permuting rows and columns will not change the condition numbers
// (in exact arithmetic) but diagonal scaling will.  For further
// explanation of balancing, see section 4.10.2 of the LAPACK
// Users' Guide.
func Zgeevx(balanc, jobvl, jobvr, sense byte, n *int, a *mat.CMatrix, lda *int, w *mat.CVector, vl *mat.CMatrix, ldvl *int, vr *mat.CMatrix, ldvr, ilo, ihi *int, scale *mat.Vector, abnrm *float64, rconde, rcondv *mat.Vector, work *mat.CVector, lwork *int, rwork *mat.Vector, info *int) {
	var lquery, scalea, wantvl, wantvr, wntsnb, wntsne, wntsnn, wntsnv bool
	var job, side byte
	var tmp complex128
	var anrm, bignum, cscale, eps, one, scl, smlnum, zero float64
	var hswork, i, icond, ierr, itau, iwrk, k, lworkTrevc, maxwrk, minwrk, nout int
	_select := make([]bool, 1)
	dum := vf(1)

	zero = 0.0
	one = 1.0

	//     Test the input arguments
	(*info) = 0
	lquery = ((*lwork) == -1)
	wantvl = jobvl == 'V'
	wantvr = jobvr == 'V'
	wntsnn = sense == 'N'
	wntsne = sense == 'E'
	wntsnv = sense == 'V'
	wntsnb = sense == 'B'
	if !(balanc == 'N' || balanc == 'S' || balanc == 'P' || balanc == 'B') {
		(*info) = -1
	} else if (!wantvl) && (jobvl != 'N') {
		(*info) = -2
	} else if (!wantvr) && (jobvr != 'N') {
		(*info) = -3
	} else if !(wntsnn || wntsne || wntsnb || wntsnv) || ((wntsne || wntsnb) && !(wantvl && wantvr)) {
		(*info) = -4
	} else if (*n) < 0 {
		(*info) = -5
	} else if (*lda) < maxint(1, *n) {
		(*info) = -7
	} else if (*ldvl) < 1 || (wantvl && (*ldvl) < (*n)) {
		(*info) = -10
	} else if (*ldvr) < 1 || (wantvr && (*ldvr) < (*n)) {
		(*info) = -12
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       CWorkspace refers to complex workspace, and RWorkspace to real
	//       workspace. NB refers to the optimal block size for the
	//       immediately following subroutine, as returned by ILAENV.
	//       HSWORK refers to the workspace preferred by ZHSEQR, as
	//       calculated below. HSWORK is computed assuming ILO=1 and IHI=N,
	//       the worst case.)
	if (*info) == 0 {
		if (*n) == 0 {
			minwrk = 1
			maxwrk = 1
		} else {
			maxwrk = (*n) + (*n)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEHRD"), []byte{' '}, n, func() *int { y := 1; return &y }(), n, func() *int { y := 0; return &y }())

			if wantvl {
				Ztrevc3('L', 'B', _select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout, work, toPtr(-1), rwork, toPtr(-1), &ierr)
				lworkTrevc = int(work.GetRe(0))
				maxwrk = maxint(maxwrk, lworkTrevc)
				Zhseqr('S', 'V', n, func() *int { y := 1; return &y }(), n, a, lda, w, vl, ldvl, work, toPtr(-1), info)
			} else if wantvr {
				Ztrevc3('R', 'B', _select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout, work, toPtr(-1), rwork, toPtr(-1), &ierr)
				lworkTrevc = int(work.GetRe(0))
				maxwrk = maxint(maxwrk, lworkTrevc)
				Zhseqr('S', 'V', n, func() *int { y := 1; return &y }(), n, a, lda, w, vr, ldvr, work, toPtr(-1), info)
			} else {
				if wntsnn {
					Zhseqr('E', 'N', n, func() *int { y := 1; return &y }(), n, a, lda, w, vr, ldvr, work, toPtr(-1), info)
				} else {
					Zhseqr('S', 'N', n, func() *int { y := 1; return &y }(), n, a, lda, w, vr, ldvr, work, toPtr(-1), info)
				}
			}
			hswork = int(work.GetRe(0))

			if (!wantvl) && (!wantvr) {
				minwrk = 2 * (*n)
				if !(wntsnn || wntsne) {
					minwrk = maxint(minwrk, (*n)*(*n)+2*(*n))
				}
				maxwrk = maxint(maxwrk, hswork)
				if !(wntsnn || wntsne) {
					maxwrk = maxint(maxwrk, (*n)*(*n)+2*(*n))
				}
			} else {
				minwrk = 2 * (*n)
				if !(wntsnn || wntsne) {
					minwrk = maxint(minwrk, (*n)*(*n)+2*(*n))
				}
				maxwrk = maxint(maxwrk, hswork)
				maxwrk = maxint(maxwrk, (*n)+((*n)-1)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNGHR"), []byte{' '}, n, func() *int { y := 1; return &y }(), n, toPtr(-1)))
				if !(wntsnn || wntsne) {
					maxwrk = maxint(maxwrk, (*n)*(*n)+2*(*n))
				}
				maxwrk = maxint(maxwrk, 2*(*n))
			}
			maxwrk = maxint(maxwrk, minwrk)
		}
		work.SetRe(0, float64(maxwrk))

		if (*lwork) < minwrk && !lquery {
			(*info) = -20
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGEEVX"), -(*info))
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
	icond = 0
	anrm = Zlange('M', n, n, a, lda, dum)
	scalea = false
	if anrm > zero && anrm < smlnum {
		scalea = true
		cscale = smlnum
	} else if anrm > bignum {
		scalea = true
		cscale = bignum
	}
	if scalea {
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &cscale, n, n, a, lda, &ierr)
	}

	//     Balance the matrix and compute ABNRM
	Zgebal(balanc, n, a, lda, ilo, ihi, scale, &ierr)
	(*abnrm) = Zlange('1', n, n, a, lda, dum)
	if scalea {
		dum.Set(0, (*abnrm))
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), &ierr)
		(*abnrm) = dum.Get(0)
	}

	//     Reduce to upper Hessenberg form
	//     (CWorkspace: need 2*N, prefer N+N*NB)
	//     (RWorkspace: none)
	itau = 1
	iwrk = itau + (*n)
	Zgehrd(n, ilo, ihi, a, lda, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &ierr)

	if wantvl {
		//        Want left eigenvectors
		//        Copy Householder vectors to VL
		side = 'L'
		Zlacpy('L', n, n, a, lda, vl, ldvl)

		//        Generate unitary matrix in VL
		//        (CWorkspace: need 2*N-1, prefer N+(N-1)*NB)
		//        (RWorkspace: none)
		Zunghr(n, ilo, ihi, vl, ldvl, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &ierr)

		//        Perform QR iteration, accumulating Schur vectors in VL
		//        (CWorkspace: need 1, prefer HSWORK (see comments) )
		//        (RWorkspace: none)
		iwrk = itau
		Zhseqr('S', 'V', n, ilo, ihi, a, lda, w, vl, ldvl, work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), info)

		if wantvr {
			//           Want left and right eigenvectors
			//           Copy Schur vectors to VR
			side = 'B'
			Zlacpy('F', n, n, vl, ldvl, vr, ldvr)
		}

	} else if wantvr {
		//        Want right eigenvectors
		//        Copy Householder vectors to VR
		side = 'R'
		Zlacpy('L', n, n, a, lda, vr, ldvr)

		//        Generate unitary matrix in VR
		//        (CWorkspace: need 2*N-1, prefer N+(N-1)*NB)
		//        (RWorkspace: none)
		Zunghr(n, ilo, ihi, vr, ldvr, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &ierr)

		//        Perform QR iteration, accumulating Schur vectors in VR
		//        (CWorkspace: need 1, prefer HSWORK (see comments) )
		//        (RWorkspace: none)
		iwrk = itau
		Zhseqr('S', 'V', n, ilo, ihi, a, lda, w, vr, ldvr, work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), info)

	} else {
		//        Compute eigenvalues only
		//        If condition numbers desired, compute Schur form
		if wntsnn {
			job = 'E'
		} else {
			job = 'S'
		}

		//        (CWorkspace: need 1, prefer HSWORK (see comments) )
		//        (RWorkspace: none)
		iwrk = itau
		Zhseqr(job, 'N', n, ilo, ihi, a, lda, w, vr, ldvr, work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), info)
	}

	//     If INFO .NE. 0 from ZHSEQR, then quit
	if (*info) != 0 {
		goto label50
	}

	if wantvl || wantvr {
		//        Compute left and/or right eigenvectors
		//        (CWorkspace: need 2*N, prefer N + 2*N*NB)
		//        (RWorkspace: need N)
		Ztrevc3(side, 'B', _select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout, work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), rwork, n, &ierr)
	}

	//     Compute condition numbers if desired
	//     (CWorkspace: need N*N+2*N unless SENSE = 'E')
	//     (RWorkspace: need 2*N unless SENSE = 'E')
	if !wntsnn {
		Ztrsna(sense, 'A', _select, n, a, lda, vl, ldvl, vr, ldvr, rconde, rcondv, n, &nout, work.CMatrixOff(iwrk-1, *n, opts), n, rwork, &icond)
	}

	if wantvl {
		//        Undo balancing of left eigenvectors
		Zgebak(balanc, 'L', n, ilo, ihi, scale, n, vl, ldvl, &ierr)

		//        Normalize left eigenvectors and make largest component real
		for i = 1; i <= (*n); i++ {
			scl = one / goblas.Dznrm2(n, vl.CVector(0, i-1), func() *int { y := 1; return &y }())
			goblas.Zdscal(n, &scl, vl.CVector(0, i-1), func() *int { y := 1; return &y }())
			for k = 1; k <= (*n); k++ {
				rwork.Set(k-1, math.Pow(vl.GetRe(k-1, i-1), 2)+math.Pow(vl.GetIm(k-1, i-1), 2))
			}
			k = goblas.Idamax(n, rwork, func() *int { y := 1; return &y }())
			tmp = vl.GetConj(k-1, i-1) / complex(math.Sqrt(rwork.Get(k-1)), 0)
			goblas.Zscal(n, &tmp, vl.CVector(0, i-1), func() *int { y := 1; return &y }())
			vl.SetRe(k-1, i-1, vl.GetRe(k-1, i-1))
		}
	}

	if wantvr {
		//        Undo balancing of right eigenvectors
		Zgebak(balanc, 'R', n, ilo, ihi, scale, n, vr, ldvr, &ierr)

		//        Normalize right eigenvectors and make largest component real
		for i = 1; i <= (*n); i++ {
			scl = one / goblas.Dznrm2(n, vr.CVector(0, i-1), func() *int { y := 1; return &y }())
			goblas.Zdscal(n, &scl, vr.CVector(0, i-1), func() *int { y := 1; return &y }())
			for k = 1; k <= (*n); k++ {
				rwork.Set(k-1, math.Pow(vr.GetRe(k-1, i-1), 2)+math.Pow(vr.GetIm(k-1, i-1), 2))
			}
			k = goblas.Idamax(n, rwork, func() *int { y := 1; return &y }())
			tmp = vr.GetConj(k-1, i-1) / complex(math.Sqrt(rwork.Get(k-1)), 0)
			goblas.Zscal(n, &tmp, vr.CVector(0, i-1), func() *int { y := 1; return &y }())
			vr.SetRe(k-1, i-1, vr.GetRe(k-1, i-1))
		}
	}

	//     Undo scaling if necessary
label50:
	;
	if scalea {
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, toPtr((*n)-(*info)), func() *int { y := 1; return &y }(), w.CMatrixOff((*info)+1-1, maxint((*n)-(*info), 1), opts), toPtr(maxint((*n)-(*info), 1)), &ierr)
		if (*info) == 0 {
			if (wntsnv || wntsnb) && icond == 0 {
				Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, n, func() *int { y := 1; return &y }(), rcondv.Matrix(*n, opts), n, &ierr)
			}
		} else {
			Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, toPtr((*ilo)-1), func() *int { y := 1; return &y }(), w.CMatrix(*n, opts), n, &ierr)
		}
	}

	work.SetRe(0, float64(maxwrk))
}
