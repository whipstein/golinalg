package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dspevd computes all the eigenvalues and, optionally, eigenvectors
// of a real symmetric matrix A in packed storage. If eigenvectors are
// desired, it uses a divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Dspevd(jobz, uplo byte, n *int, ap, w *mat.Vector, z *mat.Matrix, ldz *int, work *mat.Vector, lwork *int, iwork *[]int, liwork, info *int) {
	var lquery, wantz bool
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var iinfo, inde, indtau, indwrk, iscale, liwmin, llwork, lwmin int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'
	lquery = ((*lwork) == -1 || (*liwork) == -1)

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

	if (*info) == 0 {
		if (*n) <= 1 {
			liwmin = 1
			lwmin = 1
		} else {
			if wantz {
				liwmin = 3 + 5*(*n)
				lwmin = 1 + 6*(*n) + int(math.Pow(float64(*n), 2))
			} else {
				liwmin = 1
				lwmin = 2 * (*n)
			}
		}
		(*iwork)[0] = liwmin
		work.Set(0, float64(lwmin))

		if (*lwork) < lwmin && !lquery {
			(*info) = -9
		} else if (*liwork) < liwmin && !lquery {
			(*info) = -11
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSPEVD"), -(*info))
		return
	} else if lquery {
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
		goblas.Dscal(toPtr(((*n)*((*n)+1))/2), &sigma, ap, toPtr(1))
	}

	//     Call DSPTRD to reduce symmetric packed matrix to tridiagonal form.
	inde = 1
	indtau = inde + (*n)
	Dsptrd(uplo, n, ap, w, work.Off(inde-1), work.Off(indtau-1), &iinfo)

	//     For eigenvalues only, call DSTERF.  For eigenvectors, first call
	//     DSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
	//     tridiagonal matrix, then call DOPMTR to multiply it by the
	//     Householder transformations represented in AP.
	if !wantz {
		Dsterf(n, w, work.Off(inde-1), info)
	} else {
		indwrk = indtau + (*n)
		llwork = (*lwork) - indwrk + 1
		Dstedc('I', n, w, work.Off(inde-1), z, ldz, work.Off(indwrk-1), &llwork, iwork, liwork, info)
		Dopmtr('L', uplo, 'N', n, n, ap, work.Off(indtau-1), z, ldz, work.Off(indwrk-1), &iinfo)
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
	if iscale == 1 {
		goblas.Dscal(n, toPtrf64(one/sigma), w, toPtr(1))
	}

	work.Set(0, float64(lwmin))
	(*iwork)[0] = liwmin
}
