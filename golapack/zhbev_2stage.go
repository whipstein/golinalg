package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhbev2stage computes all the eigenvalues and, optionally, eigenvectors of
// a complex Hermitian band matrix A using the 2stage technique for
// the reduction to tridiagonal.
func Zhbev2stage(jobz, uplo byte, n, kd *int, ab *mat.CMatrix, ldab *int, w *mat.Vector, z *mat.CMatrix, ldz *int, work *mat.CVector, lwork *int, rwork *mat.Vector, info *int) {
	var lower, lquery, wantz bool
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var ib, iinfo, imax, inde, indhous, indrwk, indwrk, iscale, lhtrd, llwork, lwmin, lwtrd int

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
	} else if (*kd) < 0 {
		(*info) = -4
	} else if (*ldab) < (*kd)+1 {
		(*info) = -6
	} else if (*ldz) < 1 || (wantz && (*ldz) < (*n)) {
		(*info) = -9
	}

	if (*info) == 0 {
		if (*n) <= 1 {
			lwmin = 1
			work.SetRe(0, float64(lwmin))
		} else {
			ib = Ilaenv2stage(func() *int { y := 2; return &y }(), []byte("ZHETRD_HB2ST"), []byte{jobz}, n, kd, toPtr(-1), toPtr(-1))
			lhtrd = Ilaenv2stage(func() *int { y := 3; return &y }(), []byte("ZHETRD_HB2ST"), []byte{jobz}, n, kd, &ib, toPtr(-1))
			lwtrd = Ilaenv2stage(func() *int { y := 4; return &y }(), []byte("ZHETRD_HB2ST"), []byte{jobz}, n, kd, &ib, toPtr(-1))
			lwmin = lhtrd + lwtrd
			work.SetRe(0, float64(lwmin))
		}

		if (*lwork) < lwmin && !lquery {
			(*info) = -11
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHBEV_2STAGE "), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if (*n) == 1 {
		if lower {
			w.Set(0, ab.GetRe(0, 0))
		} else {
			w.Set(0, ab.GetRe((*kd)+1-1, 0))
		}
		if wantz {
			z.SetRe(0, 0, one)
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
	anrm = Zlanhb('M', uplo, n, kd, ab, ldab, rwork)
	iscale = 0
	if anrm > zero && anrm < rmin {
		iscale = 1
		sigma = rmin / anrm
	} else if anrm > rmax {
		iscale = 1
		sigma = rmax / anrm
	}
	if iscale == 1 {
		if lower {
			Zlascl('B', kd, kd, &one, &sigma, n, n, ab, ldab, info)
		} else {
			Zlascl('Q', kd, kd, &one, &sigma, n, n, ab, ldab, info)
		}
	}

	//     Call ZHBTRD_HB2ST to reduce Hermitian band matrix to tridiagonal form.
	inde = 1
	indhous = 1
	indwrk = indhous + lhtrd
	llwork = (*lwork) - indwrk + 1

	Zhetrdhb2st('N', jobz, uplo, n, kd, ab, ldab, w, rwork.Off(inde-1), work.Off(indhous-1), &lhtrd, work.Off(indwrk-1), &llwork, &iinfo)

	//     For eigenvalues only, call DSTERF.  For eigenvectors, call ZSTEQR.
	if !wantz {
		Dsterf(n, w, rwork.Off(inde-1), info)
	} else {
		indrwk = inde + (*n)
		Zsteqr(jobz, n, w, rwork.Off(inde-1), z, ldz, rwork.Off(indrwk-1), info)
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

	//     Set WORK(1) to optimal workspace size.
	work.SetRe(0, float64(lwmin))
}
