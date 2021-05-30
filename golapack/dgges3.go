package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dgges3 computes for a pair of N-by-N real nonsymmetric matrices (A,B),
// the generalized eigenvalues, the generalized real Schur form (S,T),
// optionally, the left and/or right matrices of Schur vectors (VSL and
// VSR). This gives the generalized Schur factorization
//
//          (A,B) = ( (VSL)*S*(VSR)**T, (VSL)*T*(VSR)**T )
//
// Optionally, it also orders the eigenvalues so that a selected cluster
// of eigenvalues appears in the leading diagonal blocks of the upper
// quasi-triangular matrix S and the upper triangular matrix T.The
// leading columns of VSL and VSR then form an orthonormal basis for the
// corresponding left and right eigenspaces (deflating subspaces).
//
// (If only the generalized eigenvalues are needed, use the driver
// DGGEV instead, which is faster.)
//
// A generalized eigenvalue for a pair of matrices (A,B) is a scalar w
// or a ratio alpha/beta = w, such that  A - w*B is singular.  It is
// usually represented as the pair (alpha,beta), as there is a
// reasonable interpretation for beta=0 or both being zero.
//
// A pair of matrices (S,T) is in generalized real Schur form if T is
// upper triangular with non-negative diagonal and S is block upper
// triangular with 1-by-1 and 2-by-2 blocks.  1-by-1 blocks correspond
// to real generalized eigenvalues, while 2-by-2 blocks of S will be
// "standardized" by making the corresponding elements of T have the
// form:
//         [  a  0  ]
//         [  0  b  ]
//
// and the pair of corresponding 2-by-2 blocks in S and T will have a
// complex conjugate pair of generalized eigenvalues.
func Dgges3(jobvsl, jobvsr, sort byte, selctg dlctesFunc, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb, sdim *int, alphar, alphai, beta *mat.Vector, vsl *mat.Matrix, ldvsl *int, vsr *mat.Matrix, ldvsr *int, work *mat.Vector, lwork *int, bwork *[]bool, info *int) {
	var cursl, ilascl, ilbscl, ilvsl, ilvsr, lastsl, lquery, lst2sl, wantst bool
	var anrm, anrmto, bignum, bnrm, bnrmto, eps, one, pvsl, pvsr, safmax, safmin, smlnum, zero float64
	var i, icols, ierr, ihi, ijobvl, ijobvr, ileft, ilo, ip, iright, irows, itau, iwrk, lwkopt int

	dif := vf(2)
	idum := make([]int, 1)

	zero = 0.0
	one = 1.0

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

	//     Test the input arguments
	(*info) = 0
	lquery = ((*lwork) == -1)
	if ijobvl <= 0 {
		(*info) = -1
	} else if ijobvr <= 0 {
		(*info) = -2
	} else if (!wantst) && sort != 'N' {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -5
	} else if (*lda) < maxint(1, *n) {
		(*info) = -7
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -9
	} else if (*ldvsl) < 1 || (ilvsl && (*ldvsl) < (*n)) {
		(*info) = -15
	} else if (*ldvsr) < 1 || (ilvsr && (*ldvsr) < (*n)) {
		(*info) = -17
	} else if (*lwork) < 6*(*n)+16 && !lquery {
		(*info) = -19
	}

	//     Compute workspace
	if (*info) == 0 {
		Dgeqrf(n, n, b, ldb, work, work, toPtr(-1), &ierr)
		lwkopt = maxint(6*(*n)+16, 3*(*n)+int(work.Get(0)))
		Dormqr('L', 'T', n, n, n, b, ldb, work, a, lda, work, toPtr(-1), &ierr)
		lwkopt = maxint(lwkopt, 3*(*n)+int(work.Get(0)))
		if ilvsl {
			Dorgqr(n, n, n, vsl, ldvsl, work, work, toPtr(-1), &ierr)
			lwkopt = maxint(lwkopt, 3*(*n)+int(work.Get(0)))
		}
		Dgghd3(jobvsl, jobvsr, n, func() *int { y := 1; return &y }(), n, a, lda, b, ldb, vsl, ldvsl, vsr, ldvsr, work, toPtr(-1), &ierr)
		lwkopt = maxint(lwkopt, 3*(*n)+int(work.Get(0)))
		Dhgeqz('S', jobvsl, jobvsr, n, func() *int { y := 1; return &y }(), n, a, lda, b, ldb, alphar, alphai, beta, vsl, ldvsl, vsr, ldvsr, work, toPtr(-1), &ierr)
		lwkopt = maxint(lwkopt, 2*(*n)+int(work.Get(0)))
		if wantst {
			Dtgsen(func() *int { y := 0; return &y }(), ilvsl, ilvsr, *bwork, n, a, lda, b, ldb, alphar, alphai, beta, vsl, ldvsl, vsr, ldvsr, sdim, &pvsl, &pvsr, dif, work, toPtr(-1), &idum, func() *int { y := 1; return &y }(), &ierr)
			lwkopt = maxint(lwkopt, 2*(*n)+int(work.Get(0)))
		}
		work.Set(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGGES3 "), -(*info))
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
	safmin = Dlamch(SafeMinimum)
	safmax = one / safmin
	Dlabad(&safmin, &safmax)
	smlnum = math.Sqrt(safmin) / eps
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

	//     Permute the matrix to make it more nearly triangular
	ileft = 1
	iright = (*n) + 1
	iwrk = iright + (*n)
	Dggbal('P', n, a, lda, b, ldb, &ilo, &ihi, work.Off(ileft-1), work.Off(iright-1), work.Off(iwrk-1), &ierr)

	//     Reduce B to triangular form (QR decomposition of B)
	irows = ihi + 1 - ilo
	icols = (*n) + 1 - ilo
	itau = iwrk
	iwrk = itau + irows
	Dgeqrf(&irows, &icols, b.Off(ilo-1, ilo-1), ldb, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)

	//     Apply the orthogonal transformation to matrix A
	Dormqr('L', 'T', &irows, &icols, &irows, b.Off(ilo-1, ilo-1), ldb, work.Off(itau-1), a.Off(ilo-1, ilo-1), lda, work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)

	//     Initialize VSL
	if ilvsl {
		Dlaset('F', n, n, &zero, &one, vsl, ldvsl)
		if irows > 1 {
			Dlacpy('L', toPtr(irows-1), toPtr(irows-1), b.Off(ilo+1-1, ilo-1), ldb, vsl.Off(ilo+1-1, ilo-1), ldvsl)
		}
		Dorgqr(&irows, &irows, &irows, vsl.Off(ilo-1, ilo-1), ldvsl, work.Off(itau-1), work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)
	}

	//     Initialize VSR
	if ilvsr {
		Dlaset('F', n, n, &zero, &one, vsr, ldvsr)
	}

	//     Reduce to generalized Hessenberg form
	Dgghd3(jobvsl, jobvsr, n, &ilo, &ihi, a, lda, b, ldb, vsl, ldvsl, vsr, ldvsr, work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)

	//     Perform QZ algorithm, computing Schur vectors if desired
	iwrk = itau
	Dhgeqz('S', jobvsl, jobvsr, n, &ilo, &ihi, a, lda, b, ldb, alphar, alphai, beta, vsl, ldvsl, vsr, ldvsr, work.Off(iwrk-1), toPtr((*lwork)+1-iwrk), &ierr)
	if ierr != 0 {
		if ierr > 0 && ierr <= (*n) {
			(*info) = ierr
		} else if ierr > (*n) && ierr <= 2*(*n) {
			(*info) = ierr - (*n)
		} else {
			(*info) = (*n) + 1
		}
		goto label50
	}

	//     Sort eigenvalues ALPHA/BETA if desired
	(*sdim) = 0
	if wantst {
		//        Undo scaling on eigenvalues before SELCTGing
		if ilascl {
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrmto, &anrm, n, func() *int { y := 1; return &y }(), alphar.Matrix(*n, opts), n, &ierr)
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrmto, &anrm, n, func() *int { y := 1; return &y }(), alphai.Matrix(*n, opts), n, &ierr)
		}
		if ilbscl {
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrmto, &bnrm, n, func() *int { y := 1; return &y }(), beta.Matrix(*n, opts), n, &ierr)
		}

		//        Select eigenvalues
		for i = 1; i <= (*n); i++ {
			(*bwork)[i-1] = selctg(alphar.GetPtr(i-1), alphai.GetPtr(i-1), beta.GetPtr(i-1))
		}

		Dtgsen(func() *int { y := 0; return &y }(), ilvsl, ilvsr, *bwork, n, a, lda, b, ldb, alphar, alphai, beta, vsl, ldvsl, vsr, ldvsr, sdim, &pvsl, &pvsr, dif, work.Off(iwrk-1), toPtr((*lwork)-iwrk+1), &idum, func() *int { y := 1; return &y }(), &ierr)
		if ierr == 1 {
			(*info) = (*n) + 3
		}

	}

	//     Apply back-permutation to VSL and VSR
	if ilvsl {
		Dggbak('P', 'L', n, &ilo, &ihi, work.Off(ileft-1), work.Off(iright-1), n, vsl, ldvsl, &ierr)
	}

	if ilvsr {
		Dggbak('P', 'R', n, &ilo, &ihi, work.Off(ileft-1), work.Off(iright-1), n, vsr, ldvsr, &ierr)
	}

	//     Check if unscaling would cause over/underflow, if so, rescale
	//     (ALPHAR(I),ALPHAI(I),BETA(I)) so BETA(I) is on the order of
	//     B(I,I) and ALPHAR(I) and ALPHAI(I) are on the order of A(I,I)
	if ilascl {
		for i = 1; i <= (*n); i++ {
			if alphai.Get(i-1) != zero {
				if (alphar.Get(i-1)/safmax) > (anrmto/anrm) || (safmin/alphar.Get(i-1)) > (anrm/anrmto) {
					work.Set(0, math.Abs(a.Get(i-1, i-1)/alphar.Get(i-1)))
					beta.Set(i-1, beta.Get(i-1)*work.Get(0))
					alphar.Set(i-1, alphar.Get(i-1)*work.Get(0))
					alphai.Set(i-1, alphai.Get(i-1)*work.Get(0))
				} else if (alphai.Get(i-1)/safmax) > (anrmto/anrm) || (safmin/alphai.Get(i-1)) > (anrm/anrmto) {
					work.Set(0, math.Abs(a.Get(i-1, i+1-1)/alphai.Get(i-1)))
					beta.Set(i-1, beta.Get(i-1)*work.Get(0))
					alphar.Set(i-1, alphar.Get(i-1)*work.Get(0))
					alphai.Set(i-1, alphai.Get(i-1)*work.Get(0))
				}
			}
		}
	}

	if ilbscl {
		for i = 1; i <= (*n); i++ {
			if alphai.Get(i-1) != zero {
				if (beta.Get(i-1)/safmax) > (bnrmto/bnrm) || (safmin/beta.Get(i-1)) > (bnrm/bnrmto) {
					work.Set(0, math.Abs(b.Get(i-1, i-1)/beta.Get(i-1)))
					beta.Set(i-1, beta.Get(i-1)*work.Get(0))
					alphar.Set(i-1, alphar.Get(i-1)*work.Get(0))
					alphai.Set(i-1, alphai.Get(i-1)*work.Get(0))
				}
			}
		}
	}

	//     Undo scaling
	if ilascl {
		Dlascl('H', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrmto, &anrm, n, n, a, lda, &ierr)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrmto, &anrm, n, func() *int { y := 1; return &y }(), alphar.Matrix(*n, opts), n, &ierr)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrmto, &anrm, n, func() *int { y := 1; return &y }(), alphai.Matrix(*n, opts), n, &ierr)
	}

	if ilbscl {
		Dlascl('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrmto, &bnrm, n, n, b, ldb, &ierr)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrmto, &bnrm, n, func() *int { y := 1; return &y }(), beta.Matrix(*n, opts), n, &ierr)
	}

	if wantst {
		//        Check if reordering is correct
		lastsl = true
		lst2sl = true
		(*sdim) = 0
		ip = 0
		for i = 1; i <= (*n); i++ {
			cursl = selctg(alphar.GetPtr(i-1), alphai.GetPtr(i-1), beta.GetPtr(i-1))
			if alphai.Get(i-1) == zero {
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

label50:
	;

	work.Set(0, float64(lwkopt))
}