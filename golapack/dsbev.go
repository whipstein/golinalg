package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsbev computes all the eigenvalues and, optionally, eigenvectors of
// a real symmetric band matrix A.
func Dsbev(jobz, uplo byte, n, kd *int, ab *mat.Matrix, ldab *int, w *mat.Vector, z *mat.Matrix, ldz *int, work *mat.Vector, info *int) {
	var lower, wantz bool
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var iinfo, imax, inde, indwrk, iscale int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'
	lower = uplo == 'L'

	(*info) = 0
	if !(wantz || jobz == 'N') {
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

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSBEV "), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if (*n) == 1 {
		if lower {
			w.Set(0, ab.Get(0, 0))
		} else {
			w.Set(0, ab.Get((*kd), 0))
		}
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
	anrm = Dlansb('M', uplo, n, kd, ab, ldab, work)
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
			Dlascl('B', kd, kd, &one, &sigma, n, n, ab, ldab, info)
		} else {
			Dlascl('Q', kd, kd, &one, &sigma, n, n, ab, ldab, info)
		}
	}

	//     Call DSBTRD to reduce symmetric band matrix to tridiagonal form.
	inde = 1
	indwrk = inde + (*n)
	Dsbtrd(jobz, uplo, n, kd, ab, ldab, w, work.Off(inde-1), z, ldz, work.Off(indwrk-1), &iinfo)

	//     For eigenvalues only, call DSTERF.  For eigenvectors, call SSTEQR.
	if !wantz {
		Dsterf(n, w, work.Off(inde-1), info)
	} else {
		Dsteqr(jobz, n, w, work.Off(inde-1), z, ldz, work.Off(indwrk-1), info)
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
	if iscale == 1 {
		if (*info) == 0 {
			imax = (*n)
		} else {
			imax = (*info) - 1
		}
		goblas.Dscal(imax, one/sigma, w.Off(0, 1))
	}
}
