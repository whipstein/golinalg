package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Zheevd computes all eigenvalues and, optionally, eigenvectors of a
// complex Hermitian matrix A.  If eigenvectors are desired, it uses a
// divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Zheevd(jobz, uplo byte, n *int, a *mat.CMatrix, lda *int, w *mat.Vector, work *mat.CVector, lwork *int, rwork *mat.Vector, lrwork *int, iwork *[]int, liwork, info *int) {
	var lower, lquery, wantz bool
	var cone complex128
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var iinfo, imax, inde, indrwk, indtau, indwk2, indwrk, iscale, liopt, liwmin, llrwk, llwork, llwrk2, lopt, lropt, lrwmin, lwmin int

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters.
	wantz = jobz == 'V'
	lower = uplo == 'L'
	lquery = ((*lwork) == -1 || (*lrwork) == -1 || (*liwork) == -1)

	(*info) = 0
	if !(wantz || jobz == 'N') {
		(*info) = -1
	} else if !(lower || uplo == 'U') {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	}

	if (*info) == 0 {
		if (*n) <= 1 {
			lwmin = 1
			lrwmin = 1
			liwmin = 1
			lopt = lwmin
			lropt = lrwmin
			liopt = liwmin
		} else {
			if wantz {
				lwmin = 2*(*n) + (*n)*(*n)
				lrwmin = 1 + 5*(*n) + 2*powint(*n, 2)
				liwmin = 3 + 5*(*n)
			} else {
				lwmin = (*n) + 1
				lrwmin = (*n)
				liwmin = 1
			}
			lopt = maxint(lwmin, (*n)+Ilaenv(func() *int { y := 1; return &y }(), []byte("ZHETRD"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1)))
			lropt = lrwmin
			liopt = liwmin
		}
		work.SetRe(0, float64(lopt))
		rwork.Set(0, float64(lropt))
		(*iwork)[0] = liopt

		if (*lwork) < lwmin && !lquery {
			(*info) = -8
		} else if (*lrwork) < lrwmin && !lquery {
			(*info) = -10
		} else if (*liwork) < liwmin && !lquery {
			(*info) = -12
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHEEVD"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if (*n) == 1 {
		w.Set(0, a.GetRe(0, 0))
		if wantz {
			a.Set(0, 0, cone)
		}
		return
	}

	//     Get machine constants.
	safmin = Dlamch(SafeMinimum)
	eps = Dlamch(Precision)
	smlnum = safmin / eps
	bignum = one / smlnum
	rmin = math.Sqrt(smlnum)
	rmax = math.Sqrt(bignum)

	//     Scale matrix to allowable range, if necessary.
	anrm = Zlanhe('M', uplo, n, a, lda, rwork)
	iscale = 0
	if anrm > zero && anrm < rmin {
		iscale = 1
		sigma = rmin / anrm
	} else if anrm > rmax {
		iscale = 1
		sigma = rmax / anrm
	}
	if iscale == 1 {
		Zlascl(uplo, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, &sigma, n, n, a, lda, info)
	}

	//     Call ZHETRD to reduce Hermitian matrix to tridiagonal form.
	inde = 1
	indtau = 1
	indwrk = indtau + (*n)
	indrwk = inde + (*n)
	indwk2 = indwrk + (*n)*(*n)
	llwork = (*lwork) - indwrk + 1
	llwrk2 = (*lwork) - indwk2 + 1
	llrwk = (*lrwork) - indrwk + 1
	Zhetrd(uplo, n, a, lda, w, rwork.Off(inde-1), work.Off(indtau-1), work.Off(indwrk-1), &llwork, &iinfo)

	//     For eigenvalues only, call DSTERF.  For eigenvectors, first call
	//     ZSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
	//     tridiagonal matrix, then call ZUNMTR to multiply it to the
	//     Householder transformations represented as Householder vectors in
	//     A.
	if !wantz {
		Dsterf(n, w, rwork.Off(inde-1), info)
	} else {
		Zstedc('I', n, w, rwork.Off(inde-1), work.CMatrixOff(indwrk-1, *n, opts), n, work.Off(indwk2-1), &llwrk2, rwork.Off(indrwk-1), &llrwk, iwork, liwork, info)
		Zunmtr('L', uplo, 'N', n, n, a, lda, work.Off(indtau-1), work.CMatrixOff(indwrk-1, *n, opts), n, work.Off(indwk2-1), &llwrk2, &iinfo)
		Zlacpy('A', n, n, work.CMatrixOff(indwrk-1, *n, opts), n, a, lda)
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
	if iscale == 1 {
		if (*info) == 0 {
			imax = (*n)
		} else {
			imax = (*info) - 1
		}
		goblas.Dscal(&imax, toPtrf64(one/sigma), w, func() *int { y := 1; return &y }())
	}

	work.SetRe(0, float64(lopt))
	rwork.Set(0, float64(lropt))
	(*iwork)[0] = liopt
}
