package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dgees computes for an N-by-N real nonsymmetric matrix A, the
// eigenvalues, the real Schur form T, and, optionally, the matrix of
// Schur vectors Z.  This gives the Schur factorization A = Z*T*(Z**T).
//
// Optionally, it also orders the eigenvalues on the diagonal of the
// real Schur form so that selected eigenvalues are at the top left.
// The leading columns of Z then form an orthonormal basis for the
// invariant subspace corresponding to the selected eigenvalues.
//
// A matrix is in real Schur form if it is upper quasi-triangular with
// 1-by-1 and 2-by-2 blocks. 2-by-2 blocks will be standardized in the
// form
//         [  a  b  ]
//         [  c  a  ]
//
// where b*c < 0. The eigenvalues of such a block are a +- math.Sqrt(bc).
func Dgees(jobvs, sort byte, _select dslectFunc, n *int, a *mat.Matrix, lda, sdim *int, wr, wi *mat.Vector, vs *mat.Matrix, ldvs *int, work *mat.Vector, lwork *int, bwork *[]bool, info *int) {
	var cursl, lastsl, lquery, lst2sl, scalea, wantst, wantvs bool
	var anrm, bignum, cscale, eps, one, s, sep, smlnum, zero float64
	var hswork, i, i1, i2, ibal, icond, ierr, ieval, ihi, ilo, inxt, ip, itau, iwrk, maxwrk, minwrk int

	dum := vf(1)
	idum := make([]int, 1)

	zero = 0.0
	one = 1.0

	//     Test the input arguments
	(*info) = 0
	lquery = ((*lwork) == -1)
	wantvs = jobvs == 'V'
	wantst = sort == 'S'
	if (!wantvs) && jobvs != 'N' {
		(*info) = -1
	} else if (!wantst) && sort != 'N' {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*lda) < maxint(1, *n) {
		(*info) = -6
	} else if (*ldvs) < 1 || (wantvs && (*ldvs) < (*n)) {
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
			minwrk = 3 * (*n)

			Dhseqr('S', jobvs, n, func() *int { y := 1; return &y }(), n, a, lda, wr, wi, vs, ldvs, work, toPtr(-1), &ieval)
			hswork = int(work.Get(0))

			if !wantvs {
				maxwrk = maxint(maxwrk, (*n)+hswork)
			} else {
				maxwrk = maxint(maxwrk, 2*(*n)+((*n)-1)*Ilaenv(func() *int { y := 1; return &y }(), []byte("DORGHR"), []byte{' '}, n, func() *int { y := 1; return &y }(), n, toPtr(-1)))
				maxwrk = maxint(maxwrk, (*n)+hswork)
			}
		}
		work.Set(0, float64(maxwrk))

		if (*lwork) < minwrk && !lquery {
			(*info) = -13
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGEES "), -(*info))
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

	//     Permute the matrix to make it more nearly triangular
	//     (Workspace: need N)
	ibal = 1
	Dgebal('P', n, a, lda, &ilo, &ihi, work.Off(ibal-1), &ierr)

	//     Reduce to upper Hessenberg form
	//     (Workspace: need 3*N, prefer 2*N+N*NB)
	itau = (*n) + ibal
	iwrk = (*n) + itau
	Dgehrd(n, &ilo, &ihi, a, lda, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &ierr)

	if wantvs {
		//        Copy Householder vectors to VS
		Dlacpy('L', n, n, a, lda, vs, ldvs)

		//        Generate orthogonal matrix in VS
		//        (Workspace: need 3*N-1, prefer 2*N+(N-1)*NB)
		Dorghr(n, &ilo, &ihi, vs, ldvs, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &ierr)
	}

	(*sdim) = 0

	//     Perform QR iteration, accumulating Schur vectors in VS if desired
	//     (Workspace: need N+1, prefer N+HSWORK (see comments) )
	iwrk = itau
	Dhseqr('S', jobvs, n, &ilo, &ihi, a, lda, wr, wi, vs, ldvs, work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &ieval)
	if ieval > 0 {
		(*info) = ieval
	}

	//     Sort eigenvalues if desired
	if wantst && (*info) == 0 {
		if scalea {
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, n, func() *int { y := 1; return &y }(), wr.Matrix(*n, opts), n, &ierr)
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, n, func() *int { y := 1; return &y }(), wi.Matrix(*n, opts), n, &ierr)
		}
		for i = 1; i <= (*n); i++ {
			(*bwork)[i-1] = _select(wr.GetPtr(i-1), wi.GetPtr(i-1))
		}

		//        Reorder eigenvalues and transform Schur vectors
		//        (Workspace: none needed)
		Dtrsen('N', jobvs, *bwork, n, a, lda, vs, ldvs, wr, wi, sdim, &s, &sep, work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &idum, func() *int { y := 1; return &y }(), &icond)
		if icond > 0 {
			(*info) = (*n) + icond
		}
	}

	if wantvs {
		//        Undo balancing
		//        (Workspace: need N)
		Dgebak('P', 'R', n, &ilo, &ihi, work.Off(ibal-1), n, vs, ldvs, &ierr)
	}

	if scalea {
		//        Undo scaling for the Schur form of A
		Dlascl('H', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, n, n, a, lda, &ierr)
		goblas.Dcopy(n, a.VectorIdx(0), toPtr((*lda)+1), wr, func() *int { y := 1; return &y }())
		if cscale == smlnum {
			//           If scaling back towards underflow, adjust WI if an
			//           offdiagonal element of a 2-by-2 block in the Schur form
			//           underflows.
			if ieval > 0 {
				i1 = ieval + 1
				i2 = ihi - 1
				Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, toPtr(ilo-1), func() *int { y := 1; return &y }(), wi.Matrix(maxint(ilo-1, 1), opts), toPtr(maxint(ilo-1, 1)), &ierr)
			} else if wantst {
				i1 = 1
				i2 = (*n) - 1
			} else {
				i1 = ilo
				i2 = ihi - 1
			}
			inxt = i1 - 1
			for i = i1; i <= i2; i++ {
				if i < inxt {
					goto label20
				}
				if wi.Get(i-1) == zero {
					inxt = i + 1
				} else {
					if a.Get(i+1-1, i-1) == zero {
						wi.Set(i-1, zero)
						wi.Set(i+1-1, zero)
					} else if a.Get(i+1-1, i-1) != zero && a.Get(i-1, i+1-1) == zero {
						wi.Set(i-1, zero)
						wi.Set(i+1-1, zero)
						if i > 1 {
							goblas.Dswap(toPtr(i-1), a.Vector(0, i-1), func() *int { y := 1; return &y }(), a.Vector(0, i+1-1), func() *int { y := 1; return &y }())
						}
						if (*n) > i+1 {
							goblas.Dswap(toPtr((*n)-i-1), a.Vector(i-1, i+2-1), lda, a.Vector(i+1-1, i+2-1), lda)
						}
						if wantvs {
							goblas.Dswap(n, vs.Vector(0, i-1), func() *int { y := 1; return &y }(), vs.Vector(0, i+1-1), func() *int { y := 1; return &y }())
						}
						a.Set(i-1, i+1-1, a.Get(i+1-1, i-1))
						a.Set(i+1-1, i-1, zero)
					}
					inxt = i + 2
				}
			label20:
			}
		}

		//        Undo scaling for the imaginary part of the eigenvalues
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &cscale, &anrm, toPtr((*n)-ieval), func() *int { y := 1; return &y }(), wi.MatrixOff(ieval+1-1, maxint((*n)-ieval, 1), opts), toPtr(maxint((*n)-ieval, 1)), &ierr)
	}

	if wantst && (*info) == 0 {
		//        Check if reordering successful
		lastsl = true
		lst2sl = true
		(*sdim) = 0
		ip = 0
		for i = 1; i <= (*n); i++ {
			cursl = _select(wr.GetPtr(i-1), wi.GetPtr(i-1))
			if wi.Get(i-1) == zero {
				if cursl {
					(*sdim) = (*sdim) + 1
				}
				ip = 0
				if cursl && !lastsl {
					(*info) = (*n) + 2
				}
			} else {
				if ip == 1 {
					//                 Last eigenvalue of conjugate pair
					cursl = cursl || lastsl
					lastsl = cursl
					if cursl {
						(*sdim) = (*sdim) + 2
					}
					ip = -1
					if cursl && !lst2sl {
						(*info) = (*n) + 2
					}
				} else {
					//                 First eigenvalue of conjugate pair
					ip = 1
				}
			}
			lst2sl = lastsl
			lastsl = cursl
		}
	}

	work.Set(0, float64(maxwrk))
}
