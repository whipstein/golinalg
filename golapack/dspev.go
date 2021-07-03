package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dspev computes all the eigenvalues and, optionally, eigenvectors of a
// real symmetric matrix A in packed storage.
func Dspev(jobz, uplo byte, n *int, ap, w *mat.Vector, z *mat.Matrix, ldz *int, work *mat.Vector, info *int) {
	var wantz bool
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var iinfo, imax, inde, indtau, indwrk, iscale int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'

	(*info) = 0
	if !(wantz || jobz == 'N') {
		(*info) = -1
	} else if !(uplo == 'U' || uplo == 'L') {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*ldz) < 1 || (wantz && (*ldz) < (*n)) {
		(*info) = -7
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSPEV "), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if (*n) == 1 {
		w.Set(0, ap.Get(0))
		if wantz {
			z.Set(0, 0, one)
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
	anrm = Dlansp('M', uplo, n, ap, work)
	iscale = 0
	if anrm > zero && anrm < rmin {
		iscale = 1
		sigma = rmin / anrm
	} else if anrm > rmax {
		iscale = 1
		sigma = rmax / anrm
	}
	if iscale == 1 {
		goblas.Dscal(((*n)*((*n)+1))/2, sigma, ap, 1)
	}

	//     Call DSPTRD to reduce symmetric packed matrix to tridiagonal form.
	inde = 1
	indtau = inde + (*n)
	Dsptrd(uplo, n, ap, w, work.Off(inde-1), work.Off(indtau-1), &iinfo)

	//     For eigenvalues only, call DSTERF.  For eigenvectors, first call
	//     DOPGTR to generate the orthogonal matrix, then call DSTEQR.
	if !wantz {
		Dsterf(n, w, work.Off(inde-1), info)
	} else {
		indwrk = indtau + (*n)
		Dopgtr(uplo, n, ap, work.Off(indtau-1), z, ldz, work.Off(indwrk-1), &iinfo)
		Dsteqr(jobz, n, w, work.Off(inde-1), z, ldz, work.Off(indtau-1), info)
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
}
