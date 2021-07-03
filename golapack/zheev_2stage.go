package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zheev2stage computes all eigenvalues and, optionally, eigenvectors of a
// complex Hermitian matrix A using the 2stage technique for
// the reduction to tridiagonal.
func Zheev2stage(jobz, uplo byte, n *int, a *mat.CMatrix, lda *int, w *mat.Vector, work *mat.CVector, lwork *int, rwork *mat.Vector, info *int) {
	var lower, lquery, wantz bool
	var cone complex128
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var ib, iinfo, imax, inde, indhous, indtau, indwrk, iscale, kd, lhtrd, llwork, lwmin, lwtrd int

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)

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
		kd = Ilaenv2stage(func() *int { y := 1; return &y }(), []byte("ZHETRD_2STAGE"), []byte{jobz}, n, toPtr(-1), toPtr(-1), toPtr(-1))
		ib = Ilaenv2stage(func() *int { y := 2; return &y }(), []byte("ZHETRD_2STAGE"), []byte{jobz}, n, &kd, toPtr(-1), toPtr(-1))
		lhtrd = Ilaenv2stage(func() *int { y := 3; return &y }(), []byte("ZHETRD_2STAGE"), []byte{jobz}, n, &kd, &ib, toPtr(-1))
		lwtrd = Ilaenv2stage(func() *int { y := 4; return &y }(), []byte("ZHETRD_2STAGE"), []byte{jobz}, n, &kd, &ib, toPtr(-1))
		lwmin = (*n) + lhtrd + lwtrd
		work.SetRe(0, float64(lwmin))

		if (*lwork) < lwmin && !lquery {
			(*info) = -8
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHEEV_2STAGE "), -(*info))
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
		work.Set(0, 1)
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

	//     Call ZHETRD_2STAGE to reduce Hermitian matrix to tridiagonal form.
	inde = 1
	indtau = 1
	indhous = indtau + (*n)
	indwrk = indhous + lhtrd
	llwork = (*lwork) - indwrk + 1

	Zhetrd2stage(jobz, uplo, n, a, lda, w, rwork.Off(inde-1), work.Off(indtau-1), work.Off(indhous-1), &lhtrd, work.Off(indwrk-1), &llwork, &iinfo)

	//     For eigenvalues only, call DSTERF.  For eigenvectors, first call
	//     ZUNGTR to generate the unitary matrix, then call ZSTEQR.
	if !wantz {
		Dsterf(n, w, rwork.Off(inde-1), info)
	} else {
		Zungtr(uplo, n, a, lda, work.Off(indtau-1), work.Off(indwrk-1), &llwork, &iinfo)
		indwrk = inde + (*n)
		Zsteqr(jobz, n, w, rwork.Off(inde-1), a, lda, rwork.Off(indwrk-1), info)
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
	if iscale == 1 {
		if (*info) == 0 {
			imax = (*n)
		} else {
			imax = (*info) - 1
		}
		goblas.Dscal(imax, one/sigma, w, 1)
	}

	//     Set WORK(1) to optimal complex workspace size.
	work.SetRe(0, float64(lwmin))
}
