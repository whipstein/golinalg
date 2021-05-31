package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsyevd2stage computes all eigenvalues and, optionally, eigenvectors of a
// real symmetric matrix A using the 2stage technique for
// the reduction to tridiagonal. If eigenvectors are desired, it uses a
// divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Dsyevd2stage(jobz, uplo byte, n *int, a *mat.Matrix, lda *int, w, work *mat.Vector, lwork *int, iwork *[]int, liwork, info *int) {
	var lower, lquery, wantz bool
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var ib, iinfo, inde, indhous, indtau, indwk2, indwrk, iscale, kd, lhtrd, liwmin, llwork, llwrk2, lwmin, lwtrd int
	_ = llwrk2

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'
	lower = uplo == 'L'
	lquery = ((*lwork) == -1 || (*liwork) == -1)

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
		if (*n) <= 1 {
			liwmin = 1
			lwmin = 1
		} else {
			kd = Ilaenv2stage(func() *int { y := 1; return &y }(), []byte("DSYTRD_2STAGE"), []byte{jobz}, n, toPtr(-1), toPtr(-1), toPtr(-1))
			ib = Ilaenv2stage(func() *int { y := 2; return &y }(), []byte("DSYTRD_2STAGE"), []byte{jobz}, n, &kd, toPtr(-1), toPtr(-1))
			lhtrd = Ilaenv2stage(func() *int { y := 3; return &y }(), []byte("DSYTRD_2STAGE"), []byte{jobz}, n, &kd, &ib, toPtr(-1))
			lwtrd = Ilaenv2stage(func() *int { y := 4; return &y }(), []byte("DSYTRD_2STAGE"), []byte{jobz}, n, &kd, &ib, toPtr(-1))
			if wantz {
				liwmin = 3 + 5*(*n)
				lwmin = 1 + 6*(*n) + 2*int(math.Pow(float64(*n), 2))
			} else {
				liwmin = 1
				lwmin = 2*(*n) + 1 + lhtrd + lwtrd
			}
		}
		work.Set(0, float64(lwmin))
		(*iwork)[0] = liwmin

		if (*lwork) < lwmin && !lquery {
			(*info) = -8
		} else if (*liwork) < liwmin && !lquery {
			(*info) = -10
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYEVD_2STAGE"), -(*info))
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
		Dlascl(uplo, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, &sigma, n, n, a, lda, info)
	}

	//     Call DSYTRD_2STAGE to reduce symmetric matrix to tridiagonal form.
	inde = 1
	indtau = inde + (*n)
	indhous = indtau + (*n)
	indwrk = indhous + lhtrd
	llwork = (*lwork) - indwrk + 1
	indwk2 = indwrk + (*n)*(*n)
	llwrk2 = (*lwork) - indwk2 + 1

	Dsytrd2stage(jobz, uplo, n, a, lda, w, work.Off(inde-1), work.Off(indtau-1), work.Off(indhous-1), &lhtrd, work.Off(indwrk-1), &llwork, &iinfo)

	//     For eigenvalues only, call DSTERF.  For eigenvectors, first call
	//     DSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
	//     tridiagonal matrix, then call DORMTR to multiply it by the
	//     Householder transformations stored in A.
	if !wantz {
		Dsterf(n, w, work.Off(inde-1), info)
	} else {
		//        Not available in this release, and argument checking should not
		//        let it getting here
		return
		// Dstedc('I', n, w, work.Off(inde-1), work.ToMatrixOff(indwrk-1, *n, opts), n, work.Off(indwk2-1), &llwrk2, iwork, liwork, info)
		// Dormtr('L', uplo, 'N', n, n, a, lda, work.Off(indtau-1), work.ToMatrixOff(indwrk-1, *n, opts), n, work.Off(indwk2-1), &llwrk2, &iinfo)
		// Dlacpy('A', n, n, work.ToMatrixOff(indwrk-1, *n, opts), n, a, lda)
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
	if iscale == 1 {
		goblas.Dscal(n, toPtrf64(one/sigma), w, func() *int { y := 1; return &y }())
	}

	work.Set(0, float64(lwmin))
	(*iwork)[0] = liwmin
}
