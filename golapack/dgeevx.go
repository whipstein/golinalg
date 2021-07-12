package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgeevx computes for an N-by-N real nonsymmetric matrix A, the
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
// where u(j)**H denotes the conjugate-transpose of u(j).
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
func Dgeevx(balanc, jobvl, jobvr, sense byte, n *int, a *mat.Matrix, lda *int, wr, wi *mat.Vector, vl *mat.Matrix, ldvl *int, vr *mat.Matrix, ldvr, ilo, ihi *int, scale *mat.Vector, abnrm *float64, rconde, rcondv, work *mat.Vector, lwork *int, iwork *[]int, info *int) {
	var lquery, scalea, wantvl, wantvr, wntsnb, wntsne, wntsnn, wntsnv bool
	var job, side byte
	var anrm, bignum, cs, cscale, eps, one, r, scl, smlnum, sn, zero float64
	var hswork, i, icond, ierr, itau, iwrk, k, lworkTrevc, maxwrk, minwrk, nout int

	_select := make([]bool, 1)
	dum := vf(1)

	zero = 0.0
	one = 1.0
	nout = 6

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
	} else if (*lda) < max(1, *n) {
		(*info) = -7
	} else if (*ldvl) < 1 || (wantvl && (*ldvl) < (*n)) {
		(*info) = -11
	} else if (*ldvr) < 1 || (wantvr && (*ldvr) < (*n)) {
		(*info) = -13
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       NB refers to the optimal block size for the immediately
	//       following subroutine, as returned by ILAENV.
	//       HSWORK refers to the workspace preferred by DHSEQR, as
	//       calculated below. HSWORK is computed assuming ILO=1 and IHI=N,
	//       the worst case.)
	if (*info) == 0 {
		if (*n) == 0 {
			minwrk = 1
			maxwrk = 1
		} else {
			maxwrk = (*n) + (*n)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DGEHRD"), []byte{' '}, n, func() *int { y := 1; return &y }(), n, func() *int { y := 0; return &y }())

			if wantvl {
				Dtrevc3('L', 'B', &_select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout, work, toPtr(-1), &ierr)
				lworkTrevc = int(work.Get(0))
				maxwrk = max(maxwrk, (*n)+lworkTrevc)
				Dhseqr('S', 'V', n, func() *int { y := 1; return &y }(), n, a, lda, wr, wi, vl, ldvl, work, toPtr(-1), info)
			} else if wantvr {
				Dtrevc3('R', 'B', &_select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout, work, toPtr(-1), &ierr)
				lworkTrevc = int(work.Get(0))
				maxwrk = max(maxwrk, (*n)+lworkTrevc)
				Dhseqr('S', 'V', n, func() *int { y := 1; return &y }(), n, a, lda, wr, wi, vr, ldvr, work, toPtr(-1), info)
			} else {
				if wntsnn {
					Dhseqr('E', 'N', n, func() *int { y := 1; return &y }(), n, a, lda, wr, wi, vr, ldvr, work, toPtr(-1), info)
				} else {
					Dhseqr('S', 'N', n, func() *int { y := 1; return &y }(), n, a, lda, wr, wi, vr, ldvr, work, toPtr(-1), info)
				}
			}
			hswork = int(work.Get(0))

			if (!wantvl) && (!wantvr) {
				minwrk = 2 * (*n)
				if !wntsnn {
					minwrk = max(minwrk, (*n)*(*n)+6*(*n))
				}
				maxwrk = max(maxwrk, hswork)
				if !wntsnn {
					maxwrk = max(maxwrk, (*n)*(*n)+6*(*n))
				}
			} else {
				minwrk = 3 * (*n)
				if (!wntsnn) && (!wntsne) {
					minwrk = max(minwrk, (*n)*(*n)+6*(*n))
				}
				maxwrk = max(maxwrk, hswork)
				maxwrk = max(maxwrk, (*n)+((*n)-1)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DORGHR"), []byte{' '}, n, func() *int { y := 1; return &y }(), n, toPtr(-1)))
				if (!wntsnn) && (!wntsne) {
					maxwrk = max(maxwrk, (*n)*(*n)+6*(*n))
				}
				maxwrk = max(maxwrk, 3*(*n))
			}
			maxwrk = max(maxwrk, minwrk)
		}
		work.Set(0, float64(maxwrk))

		if (*lwork) < minwrk && !lquery {
			(*info) = -21
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGEEVX"), -(*info))
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
	icond = 0
	anrm = Dlange('M', n, n, a, lda, dum)
	scalea = false
	if anrm > zero && anrm < smlnum {
		scalea = true
		cscale = smlnum
	} else if anrm > bignum {
		scalea = true
		cscale = bignum
	}
	if scalea {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &cscale, n, n, a, lda, &ierr)
	}

	//     Balance the matrix and compute ABNRM
	Dgebal(balanc, n, a, lda, ilo, ihi, scale, &ierr)
	(*abnrm) = Dlange('1', n, n, a, lda, dum)
	if scalea {
		dum.Set(0, *abnrm)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), dum.Matrix(1, opts), func() *int { y := 1; return &y }(), &ierr)
		(*abnrm) = dum.Get(0)
	}

	//     Reduce to upper Hessenberg form
	//     (Workspace: need 2*N, prefer N+N*NB)
	itau = 1
	iwrk = itau + (*n)
	Dgehrd(n, ilo, ihi, a, lda, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &ierr)

	if wantvl {
		//        Want left eigenvectors
		//        Copy Householder vectors to VL
		side = 'L'
		Dlacpy('L', n, n, a, lda, vl, ldvl)

		//        Generate orthogonal matrix in VL
		//        (Workspace: need 2*N-1, prefer N+(N-1)*NB)
		Dorghr(n, ilo, ihi, vl, ldvl, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &ierr)

		//        Perform QR iteration, accumulating Schur vectors in VL
		//        (Workspace: need 1, prefer HSWORK (see comments) )
		iwrk = itau
		Dhseqr('S', 'V', n, ilo, ihi, a, lda, wr, wi, vl, ldvl, work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), info)

		if wantvr {
			//           Want left and right eigenvectors
			//           Copy Schur vectors to VR
			side = 'B'
			Dlacpy('F', n, n, vl, ldvl, vr, ldvr)
		}

	} else if wantvr {
		//        Want right eigenvectors
		//        Copy Householder vectors to VR
		side = 'R'
		Dlacpy('L', n, n, a, lda, vr, ldvr)

		//        Generate orthogonal matrix in VR
		//        (Workspace: need 2*N-1, prefer N+(N-1)*NB)
		Dorghr(n, ilo, ihi, vr, ldvr, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &ierr)

		//        Perform QR iteration, accumulating Schur vectors in VR
		//        (Workspace: need 1, prefer HSWORK (see comments) )
		iwrk = itau
		Dhseqr('S', 'V', n, ilo, ihi, a, lda, wr, wi, vr, ldvr, work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), info)

	} else {
		//        Compute eigenvalues only
		//        If condition numbers desired, compute Schur form
		if wntsnn {
			job = 'E'
		} else {
			job = 'S'
		}

		//        (Workspace: need 1, prefer HSWORK (see comments) )
		iwrk = itau
		Dhseqr(job, 'N', n, ilo, ihi, a, lda, wr, wi, vr, ldvr, work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), info)
	}

	//     If INFO .NE. 0 from DHSEQR, then quit
	if (*info) != 0 {
		goto label50
	}

	if wantvl || wantvr {
		//        Compute left and/or right eigenvectors
		//        (Workspace: need 3*N, prefer N + 2*N*NB)
		Dtrevc3(side, 'B', &_select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout, work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &ierr)
	}

	//     Compute condition numbers if desired
	//     (Workspace: need N*N+6*N unless SENSE = 'E')
	if !wntsnn {
		Dtrsna(sense, 'A', _select, n, a, lda, vl, ldvl, vr, ldvr, rconde, rcondv, n, &nout, work.MatrixOff(iwrk-1, *n, opts), n, iwork, &icond)
	}

	if wantvl {
		//        Undo balancing of left eigenvectors
		Dgebak(balanc, 'L', n, ilo, ihi, scale, n, vl, ldvl, &ierr)

		//        Normalize left eigenvectors and make largest component real
		for i = 1; i <= (*n); i++ {
			if wi.Get(i-1) == zero {
				scl = one / goblas.Dnrm2(*n, vl.Vector(0, i-1, 1))
				goblas.Dscal(*n, scl, vl.Vector(0, i-1, 1))
			} else if wi.Get(i-1) > zero {
				scl = one / Dlapy2(toPtrf64(goblas.Dnrm2(*n, vl.Vector(0, i-1, 1))), toPtrf64(goblas.Dnrm2(*n, vl.Vector(0, i, 1))))
				goblas.Dscal(*n, scl, vl.Vector(0, i-1, 1))
				goblas.Dscal(*n, scl, vl.Vector(0, i, 1))
				for k = 1; k <= (*n); k++ {
					work.Set(k-1, math.Pow(vl.Get(k-1, i-1), 2)+math.Pow(vl.Get(k-1, i), 2))
				}
				k = goblas.Idamax(*n, work)
				Dlartg(vl.GetPtr(k-1, i-1), vl.GetPtr(k-1, i), &cs, &sn, &r)
				goblas.Drot(*n, vl.Vector(0, i-1, 1), vl.Vector(0, i, 1), cs, sn)
				vl.Set(k-1, i, zero)
			}
		}
	}

	if wantvr {
		//        Undo balancing of right eigenvectors
		Dgebak(balanc, 'R', n, ilo, ihi, scale, n, vr, ldvr, &ierr)

		//        Normalize right eigenvectors and make largest component real
		for i = 1; i <= (*n); i++ {
			if wi.Get(i-1) == zero {
				scl = one / goblas.Dnrm2(*n, vr.Vector(0, i-1, 1))
				goblas.Dscal(*n, scl, vr.Vector(0, i-1, 1))
			} else if wi.Get(i-1) > zero {
				scl = one / Dlapy2(toPtrf64(goblas.Dnrm2(*n, vr.Vector(0, i-1, 1))), toPtrf64(goblas.Dnrm2(*n, vr.Vector(0, i, 1))))
				goblas.Dscal(*n, scl, vr.Vector(0, i-1, 1))
				goblas.Dscal(*n, scl, vr.Vector(0, i, 1))
				for k = 1; k <= (*n); k++ {
					work.Set(k-1, math.Pow(vr.Get(k-1, i-1), 2)+math.Pow(vr.Get(k-1, i), 2))
				}
				k = goblas.Idamax(*n, work)
				Dlartg(vr.GetPtr(k-1, i-1), vr.GetPtr(k-1, i), &cs, &sn, &r)
				goblas.Drot(*n, vr.Vector(0, i-1, 1), vr.Vector(0, i, 1), cs, sn)
				vr.Set(k-1, i, zero)
			}
		}
	}

	//     Undo scaling if necessary
label50:
	;
	if scalea {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, toPtr((*n)-(*info)), func() *int { y := 1; return &y }(), wr.MatrixOff((*info), max((*n)-(*info), 1), opts), toPtr(max((*n)-(*info), 1)), &ierr)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, toPtr((*n)-(*info)), func() *int { y := 1; return &y }(), wi.MatrixOff((*info), max((*n)-(*info), 1), opts), toPtr(max((*n)-(*info), 1)), &ierr)
		if (*info) == 0 {
			if (wntsnv || wntsnb) && icond == 0 {
				Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, n, func() *int { y := 1; return &y }(), rcondv.Matrix(*n, opts), n, &ierr)
			}
		} else {
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, toPtr((*ilo)-1), func() *int { y := 1; return &y }(), wr.Matrix(*n, opts), n, &ierr)
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, toPtr((*ilo)-1), func() *int { y := 1; return &y }(), wi.Matrix(*n, opts), n, &ierr)
		}
	}

	work.Set(0, float64(maxwrk))
}
