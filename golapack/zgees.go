package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Zgees computes for an N-by-N complex nonsymmetric matrix A, the
// eigenvalues, the Schur form T, and, optionally, the matrix of Schur
// vectors Z.  This gives the Schur factorization A = Z*T*(Z**H).
//
// Optionally, it also orders the eigenvalues on the diagonal of the
// Schur form so that selected eigenvalues are at the top left.
// The leading columns of Z then form an orthonormal basis for the
// invariant subspace corresponding to the selected eigenvalues.
//
// A complex matrix is in Schur form if it is upper triangular.
func Zgees(jobvs, sort byte, _select func(complex128) bool, n *int, a *mat.CMatrix, lda, sdim *int, w *mat.CVector, vs *mat.CMatrix, ldvs *int, work *mat.CVector, lwork *int, rwork *mat.Vector, bwork *[]bool, info *int) {
	var lquery, scalea, wantst, wantvs bool
	var anrm, bignum, cscale, eps, one, s, sep, smlnum, zero float64
	var hswork, i, ibal, icond, ierr, ieval, ihi, ilo, itau, iwrk, maxwrk, minwrk int
	dum := vf(1)

	zero = 0.0
	one = 1.0

	//     Test the input arguments
	(*info) = 0
	lquery = ((*lwork) == -1)
	wantvs = jobvs == 'V'
	wantst = sort == 'S'
	if (!wantvs) && (jobvs != 'N') {
		(*info) = -1
	} else if (!wantst) && (sort != 'N') {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*lda) < maxint(1, *n) {
		(*info) = -6
	} else if (*ldvs) < 1 || (wantvs && (*ldvs) < (*n)) {
		(*info) = -10
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
			minwrk = 2 * (*n)

			Zhseqr('S', jobvs, n, func() *int { y := 1; return &y }(), n, a, lda, w, vs, ldvs, work, toPtr(-1), &ieval)
			hswork = int(work.GetRe(0))

			if !wantvs {
				maxwrk = maxint(maxwrk, hswork)
			} else {
				maxwrk = maxint(maxwrk, (*n)+((*n)-1)*Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNGHR"), []byte{' '}, n, func() *int { y := 1; return &y }(), n, toPtr(-1)))
				maxwrk = maxint(maxwrk, hswork)
			}
		}
		work.SetRe(0, float64(maxwrk))

		if (*lwork) < minwrk && !lquery {
			(*info) = -12
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGEES "), -(*info))
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

	//     Scale A if maxint element outside range [SMLNUM,BIGNUM]
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

	//     Permute the matrix to make it more nearly triangular
	//     (CWorkspace: none)
	//     (RWorkspace: need N)
	ibal = 1
	Zgebal('P', n, a, lda, &ilo, &ihi, rwork.Off(ibal-1), &ierr)

	//     Reduce to upper Hessenberg form
	//     (CWorkspace: need 2*N, prefer N+N*NB)
	//     (RWorkspace: none)
	itau = 1
	iwrk = (*n) + itau
	Zgehrd(n, &ilo, &ihi, a, lda, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &ierr)

	if wantvs {
		//        Copy Householder vectors to VS
		Zlacpy('L', n, n, a, lda, vs, ldvs)

		//        Generate unitary matrix in VS
		//        (CWorkspace: need 2*N-1, prefer N+(N-1)*NB)
		//        (RWorkspace: none)
		Zunghr(n, &ilo, &ihi, vs, ldvs, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &ierr)
	}

	(*sdim) = 0

	//     Perform QR iteration, accumulating Schur vectors in VS if desired
	//     (CWorkspace: need 1, prefer HSWORK (see comments) )
	//     (RWorkspace: none)
	iwrk = itau
	Zhseqr('S', jobvs, n, &ilo, &ihi, a, lda, w, vs, ldvs, work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &ieval)
	if ieval > 0 {
		(*info) = ieval
	}

	//     Sort eigenvalues if desired
	if wantst && (*info) == 0 {
		if scalea {
			Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, n, func() *int { y := 1; return &y }(), w.CMatrix(*n, opts), n, &ierr)
		}
		for i = 1; i <= (*n); i++ {
			(*bwork)[i-1] = _select(w.Get(i - 1))
		}

		//        Reorder eigenvalues and transform Schur vectors
		//        (CWorkspace: none)
		//        (RWorkspace: none)
		Ztrsen('N', jobvs, *bwork, n, a, lda, vs, ldvs, w, sdim, &s, &sep, work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &icond)
	}

	if wantvs {
		//        Undo balancing
		//        (CWorkspace: none)
		//        (RWorkspace: need N)
		Zgebak('P', 'R', n, &ilo, &ihi, rwork.Off(ibal-1), n, vs, ldvs, &ierr)
	}

	if scalea {
		//        Undo scaling for the Schur form of A
		Zlascl('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, n, n, a, lda, &ierr)
		goblas.Zcopy(n, a.CVector(0, 0), toPtr((*lda)+1), w, func() *int { y := 1; return &y }())
	}

	work.SetRe(0, float64(maxwrk))
}
