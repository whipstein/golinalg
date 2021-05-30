package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dsyev2stage computes all eigenvalues and, optionally, eigenvectors of a
// real symmetric matrix A using the 2stage technique for
// the reduction to tridiagonal.
func Dsyev2stage(jobz, uplo byte, n *int, a *mat.Matrix, lda *int, w, work *mat.Vector, lwork, info *int) {
	var lower, lquery, wantz bool
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var ib, iinfo, imax, inde, indhous, indtau, indwrk, iscale, kd, lhtrd, llwork, lwmin, lwtrd int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'
	lower = uplo == 'L'
	lquery = ((*lwork) == -1)

	(*info) = 0
	if jobz != 'N' {
		(*info) = -1
	} else if !(lower || uplo == 'U') {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	}

	if (*info) == 0 {
		kd = Ilaenv2stage(toPtr(1), []byte("DSYTRD_2STAGE"), []byte{jobz}, n, toPtr(-1), toPtr(-1), toPtr(-1))
		ib = Ilaenv2stage(toPtr(2), []byte("DSYTRD_2STAGE"), []byte{jobz}, n, &kd, toPtr(-1), toPtr(-1))
		lhtrd = Ilaenv2stage(toPtr(3), []byte("DSYTRD_2STAGE"), []byte{jobz}, n, &kd, &ib, toPtr(-1))
		lwtrd = Ilaenv2stage(toPtr(4), []byte("DSYTRD_2STAGE"), []byte{jobz}, n, &kd, &ib, toPtr(-1))
		lwmin = 2*(*n) + lhtrd + lwtrd
		work.Set(0, float64(lwmin))

		if (*lwork) < lwmin && !lquery {
			(*info) = -8
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYEV_2STAGE "), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if (*n) == 1 {
		w.Set(0, a.Get(0, 0))
		work.Set(0, 2)
		if wantz {
			a.Set(0, 0, one)
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
	anrm = Dlansy('M', uplo, n, a, lda, work)
	iscale = 0
	if anrm > zero && anrm < rmin {
		iscale = 1
		sigma = rmin / anrm
	} else if anrm > rmax {
		iscale = 1
		sigma = rmax / anrm
	}
	if iscale == 1 {
		Dlascl(uplo, toPtr(0), toPtr(0), &one, &sigma, n, n, a, lda, info)
	}

	//     Call DSYTRD_2STAGE to reduce symmetric matrix to tridiagonal form.
	inde = 1
	indtau = inde + (*n)
	indhous = indtau + (*n)
	indwrk = indhous + lhtrd
	llwork = (*lwork) - indwrk + 1

	Dsytrd2stage(jobz, uplo, n, a, lda, w, work.Off(inde-1), work.Off(indtau-1), work.Off(indhous-1), &lhtrd, work.Off(indwrk-1), &llwork, &iinfo)

	//     For eigenvalues only, call DSTERF.  For eigenvectors, first call
	//     DORGTR to generate the orthogonal matrix, then call DSTEQR.
	if !wantz {
		Dsterf(n, w, work.Off(inde-1), info)
	} else {
		//        Not available in this release, and argument checking should not
		//        let it getting here
		return
		// Dorgtr(uplo, n, a, lda, work.Off(indtau-1), work.Off(indwrk-1), &llwork, &iinfo)
		// Dsteqr(jobz, n, w, work.Off(inde-1), a, lda, work.Off(indtau-1), info)
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
	if iscale == 1 {
		if (*info) == 0 {
			imax = (*n)
		} else {
			imax = (*info) - 1
		}
		goblas.Dscal(&imax, toPtrf64(one/sigma), w, toPtr(1))
	}

	//     Set WORK(1) to optimal workspace size.
	work.Set(0, float64(lwmin))
}
