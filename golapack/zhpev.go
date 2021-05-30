package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Zhpev computes all the eigenvalues and, optionally, eigenvectors of a
// complex Hermitian matrix in packed storage.
func Zhpev(jobz, uplo byte, n *int, ap *mat.CVector, w *mat.Vector, z *mat.CMatrix, ldz *int, work *mat.CVector, rwork *mat.Vector, info *int) {
	var wantz bool
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var iinfo, imax, inde, indrwk, indtau, indwrk, iscale int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'

	(*info) = 0
	if !(wantz || jobz == 'N') {
		(*info) = -1
	} else if !(uplo == 'L' || uplo == 'U') {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*ldz) < 1 || (wantz && (*ldz) < (*n)) {
		(*info) = -7
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHPEV "), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if (*n) == 1 {
		w.Set(0, ap.GetRe(0))
		rwork.Set(0, 1)
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
	anrm = Zlanhp('M', uplo, n, ap, rwork)
	iscale = 0
	if anrm > zero && anrm < rmin {
		iscale = 1
		sigma = rmin / anrm
	} else if anrm > rmax {
		iscale = 1
		sigma = rmax / anrm
	}
	if iscale == 1 {
		goblas.Zdscal(toPtr(((*n)*((*n)+1))/2), &sigma, ap, func() *int { y := 1; return &y }())
	}

	//     Call ZHPTRD to reduce Hermitian packed matrix to tridiagonal form.
	inde = 1
	indtau = 1
	Zhptrd(uplo, n, ap, w, rwork.Off(inde-1), work.Off(indtau-1), &iinfo)

	//     For eigenvalues only, call DSTERF.  For eigenvectors, first call
	//     ZUPGTR to generate the orthogonal matrix, then call ZSTEQR.
	if !wantz {
		Dsterf(n, w, rwork.Off(inde-1), info)
	} else {
		indwrk = indtau + (*n)
		Zupgtr(uplo, n, ap, work.Off(indtau-1), z, ldz, work.Off(indwrk-1), &iinfo)
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
}
