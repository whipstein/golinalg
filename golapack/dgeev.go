package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgeev computes for an N-by-N real nonsymmetric matrix A, the
// eigenvalues and, optionally, the left and/or right eigenvectors.
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
func Dgeev(jobvl, jobvr byte, n *int, a *mat.Matrix, lda *int, wr, wi *mat.Vector, vl *mat.Matrix, ldvl *int, vr *mat.Matrix, ldvr *int, work *mat.Vector, lwork, info *int) {
	var lquery, scalea, wantvl, wantvr bool
	var side byte
	var anrm, bignum, cs, cscale, eps, one, r, scl, smlnum, sn, zero float64
	var hswork, i, ibal, ierr, ihi, ilo, itau, iwrk, k, lworkTrevc, maxwrk, minwrk, nout int

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
	if (!wantvl) && jobvl != 'N' {
		(*info) = -1
	} else if (!wantvr) && jobvr != 'N' {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*ldvl) < 1 || (wantvl && (*ldvl) < (*n)) {
		(*info) = -9
	} else if (*ldvr) < 1 || (wantvr && (*ldvr) < (*n)) {
		(*info) = -11
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
			maxwrk = 2*(*n) + (*n)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DGEHRD"), []byte{' '}, n, func() *int { y := 1; return &y }(), n, func() *int { y := 0; return &y }())
			if wantvl {
				minwrk = 4 * (*n)
				maxwrk = maxint(maxwrk, 2*(*n)+((*n)-1)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DORGHR"), []byte{' '}, n, func() *int { y := 1; return &y }(), n, toPtr(-1)))
				Dhseqr('S', 'V', n, func() *int { y := 1; return &y }(), n, a, lda, wr, wi, vl, ldvl, work, toPtr(-1), info)
				hswork = int(work.Get(0))
				maxwrk = maxint(maxwrk, (*n)+1, (*n)+hswork)
				Dtrevc3('L', 'B', &_select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout, work, toPtr(-1), &ierr)
				lworkTrevc = int(work.Get(0))
				maxwrk = maxint(maxwrk, (*n)+lworkTrevc)
				maxwrk = maxint(maxwrk, 4*(*n))
			} else if wantvr {
				minwrk = 4 * (*n)
				maxwrk = maxint(maxwrk, 2*(*n)+((*n)-1)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DORGHR"), []byte{' '}, n, func() *int { y := 1; return &y }(), n, toPtr(-1)))
				Dhseqr('S', 'V', n, func() *int { y := 1; return &y }(), n, a, lda, wr, wi, vr, ldvr, work, toPtr(-1), info)
				hswork = int(work.Get(0))
				maxwrk = maxint(maxwrk, (*n)+1, (*n)+hswork)
				Dtrevc3('R', 'B', &_select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout, work, toPtr(-1), &ierr)
				lworkTrevc = int(work.Get(0))
				maxwrk = maxint(maxwrk, (*n)+lworkTrevc)
				maxwrk = maxint(maxwrk, 4*(*n))
			} else {
				minwrk = 3 * (*n)
				Dhseqr('E', 'N', n, func() *int { y := 1; return &y }(), n, a, lda, wr, wi, vr, ldvr, work, toPtr(-1), info)
				hswork = int(work.Get(0))
				maxwrk = maxint(maxwrk, (*n)+1, (*n)+hswork)
			}
			maxwrk = maxint(maxwrk, minwrk)
		}
		work.Set(0, float64(maxwrk))

		if (*lwork) < minwrk && !lquery {
			(*info) = -13
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGEEV "), -(*info))
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

	//     Balance the matrix
	//     (Workspace: need N)
	ibal = 1
	Dgebal('B', n, a, lda, &ilo, &ihi, work.Off(ibal-1), &ierr)

	//     Reduce to upper Hessenberg form
	//     (Workspace: need 3*N, prefer 2*N+N*NB)
	itau = ibal + (*n)
	iwrk = itau + (*n)
	Dgehrd(n, &ilo, &ihi, a, lda, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &ierr)

	if wantvl {
		//        Want left eigenvectors
		//        Copy Householder vectors to VL
		side = 'L'
		Dlacpy('L', n, n, a, lda, vl, ldvl)

		//        Generate orthogonal matrix in VL
		//        (Workspace: need 3*N-1, prefer 2*N+(N-1)*NB)
		Dorghr(n, &ilo, &ihi, vl, ldvl, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &ierr)

		//        Perform QR iteration, accumulating Schur vectors in VL
		//        (Workspace: need N+1, prefer N+HSWORK (see comments) )
		iwrk = itau
		Dhseqr('S', 'V', n, &ilo, &ihi, a, lda, wr, wi, vl, ldvl, work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), info)

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
		//        (Workspace: need 3*N-1, prefer 2*N+(N-1)*NB)
		Dorghr(n, &ilo, &ihi, vr, ldvr, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &ierr)

		//        Perform QR iteration, accumulating Schur vectors in VR
		//        (Workspace: need N+1, prefer N+HSWORK (see comments) )
		iwrk = itau
		Dhseqr('S', 'V', n, &ilo, &ihi, a, lda, wr, wi, vr, ldvr, work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), info)

	} else {
		//        Compute eigenvalues only
		//        (Workspace: need N+1, prefer N+HSWORK (see comments) )
		iwrk = itau
		Dhseqr('E', 'N', n, &ilo, &ihi, a, lda, wr, wi, vr, ldvr, work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), info)
	}

	//     If INFO .NE. 0 from DHSEQR, then quit
	if (*info) != 0 {
		goto label50
	}

	if wantvl || wantvr {
		//        Compute left and/or right eigenvectors
		//        (Workspace: need 4*N, prefer N + N + 2*N*NB)
		Dtrevc3(side, 'B', &_select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout, work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &ierr)
	}

	if wantvl {
		//        Undo balancing of left eigenvectors
		//        (Workspace: need N)
		Dgebak('B', 'L', n, &ilo, &ihi, work.Off(ibal-1), n, vl, ldvl, &ierr)

		//        Normalize left eigenvectors and make largest component real
		for i = 1; i <= (*n); i++ {
			if wi.Get(i-1) == zero {
				scl = one / goblas.Dnrm2(n, vl.Vector(0, i-1), func() *int { y := 1; return &y }())
				goblas.Dscal(n, &scl, vl.Vector(0, i-1), func() *int { y := 1; return &y }())
			} else if wi.Get(i-1) > zero {
				scl = one / Dlapy2(toPtrf64(goblas.Dnrm2(n, vl.Vector(0, i-1), func() *int { y := 1; return &y }())), toPtrf64(goblas.Dnrm2(n, vl.Vector(0, i+1-1), func() *int { y := 1; return &y }())))
				goblas.Dscal(n, &scl, vl.Vector(0, i-1), func() *int { y := 1; return &y }())
				goblas.Dscal(n, &scl, vl.Vector(0, i+1-1), func() *int { y := 1; return &y }())
				for k = 1; k <= (*n); k++ {
					work.Set(iwrk+k-1-1, math.Pow(vl.Get(k-1, i-1), 2)+math.Pow(vl.Get(k-1, i+1-1), 2))
				}
				k = goblas.Idamax(n, work.Off(iwrk-1), func() *int { y := 1; return &y }())
				Dlartg(vl.GetPtr(k-1, i-1), vl.GetPtr(k-1, i+1-1), &cs, &sn, &r)
				goblas.Drot(n, vl.Vector(0, i-1), func() *int { y := 1; return &y }(), vl.Vector(0, i+1-1), func() *int { y := 1; return &y }(), &cs, &sn)
				vl.Set(k-1, i+1-1, zero)
			}
		}
	}

	if wantvr {
		//        Undo balancing of right eigenvectors
		//        (Workspace: need N)
		Dgebak('B', 'R', n, &ilo, &ihi, work.Off(ibal-1), n, vr, ldvr, &ierr)

		//        Normalize right eigenvectors and make largest component real
		for i = 1; i <= (*n); i++ {
			if wi.Get(i-1) == zero {
				scl = one / goblas.Dnrm2(n, vr.Vector(0, i-1), func() *int { y := 1; return &y }())
				goblas.Dscal(n, &scl, vr.Vector(0, i-1), func() *int { y := 1; return &y }())
			} else if wi.Get(i-1) > zero {
				scl = one / Dlapy2(toPtrf64(goblas.Dnrm2(n, vr.Vector(0, i-1), func() *int { y := 1; return &y }())), toPtrf64(goblas.Dnrm2(n, vr.Vector(0, i+1-1), func() *int { y := 1; return &y }())))
				goblas.Dscal(n, &scl, vr.Vector(0, i-1), func() *int { y := 1; return &y }())
				goblas.Dscal(n, &scl, vr.Vector(0, i+1-1), func() *int { y := 1; return &y }())
				for k = 1; k <= (*n); k++ {
					work.Set(iwrk+k-1-1, math.Pow(vr.Get(k-1, i-1), 2)+math.Pow(vr.Get(k-1, i+1-1), 2))
				}
				k = goblas.Idamax(n, work.Off(iwrk-1), func() *int { y := 1; return &y }())
				Dlartg(vr.GetPtr(k-1, i-1), vr.GetPtr(k-1, i+1-1), &cs, &sn, &r)
				goblas.Drot(n, vr.Vector(0, i-1), func() *int { y := 1; return &y }(), vr.Vector(0, i+1-1), func() *int { y := 1; return &y }(), &cs, &sn)
				vr.Set(k-1, i+1-1, zero)
			}
		}
	}

	//     Undo scaling if necessary
label50:
	;
	if scalea {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, toPtr((*n)-(*info)), func() *int { y := 1; return &y }(), wr.MatrixOff((*info)+1-1, maxint((*n)-(*info), 1), opts), toPtr(maxint((*n)-(*info), 1)), &ierr)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, toPtr((*n)-(*info)), func() *int { y := 1; return &y }(), wi.MatrixOff((*info)+1-1, maxint((*n)-(*info), 1), opts), toPtr(maxint((*n)-(*info), 1)), &ierr)
		if (*info) > 0 {
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, toPtr(ilo-1), func() *int { y := 1; return &y }(), wr.Matrix(*n, opts), n, &ierr)
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, toPtr(ilo-1), func() *int { y := 1; return &y }(), wi.Matrix(*n, opts), n, &ierr)
		}
	}

	work.Set(0, float64(maxwrk))
}
