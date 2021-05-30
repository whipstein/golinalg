package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dsbev2stage computes all the eigenvalues and, optionally, eigenvectors of
// a real symmetric band matrix A using the 2stage technique for
// the reduction to tridiagonal.
func Dsbev2stage(jobz, uplo byte, n, kd *int, ab *mat.Matrix, ldab *int, w *mat.Vector, z *mat.Matrix, ldz *int, work *mat.Vector, lwork, info *int) {
	var lower, lquery, wantz bool
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var ib, iinfo, imax, inde, indhous, indwrk, iscale, lhtrd, llwork, lwmin, lwtrd int

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
			work.Set(0, float64(lwmin))
		} else {
			ib = Ilaenv2stage(toPtr(2), []byte("DSYTRD_SB2ST"), []byte{jobz}, n, kd, toPtr(-1), toPtr(-1))
			lhtrd = Ilaenv2stage(toPtr(3), []byte("DSYTRD_SB2ST"), []byte{jobz}, n, kd, &ib, toPtr(-1))
			lwtrd = Ilaenv2stage(toPtr(4), []byte("DSYTRD_SB2ST"), []byte{jobz}, n, kd, &ib, toPtr(-1))
			lwmin = (*n) + lhtrd + lwtrd
			work.Set(0, float64(lwmin))
		}

		if (*lwork) < lwmin && !lquery {
			(*info) = -11
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSBEV_2STAGE "), -(*info))
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
			w.Set(0, ab.Get(0, 0))
		} else {
			w.Set(0, ab.Get((*kd)+1-1, 0))
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

	//     Call DSYTRD_SB2ST to reduce symmetric band matrix to tridiagonal form.
	inde = 1
	indhous = inde + (*n)
	indwrk = indhous + lhtrd
	llwork = (*lwork) - indwrk + 1

	DsytrdSb2st('N', jobz, uplo, n, kd, ab, ldab, w, work.Off(inde-1), work.Off(indhous-1), &lhtrd, work.Off(indwrk-1), &llwork, &iinfo)

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
		goblas.Dscal(&imax, toPtrf64(one/sigma), w, toPtr(1))
	}

	//     Set WORK(1) to optimal workspace size.
	work.Set(0, float64(lwmin))
}
