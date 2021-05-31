package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhpevd computes all the eigenvalues and, optionally, eigenvectors of
// a complex Hermitian matrix A in packed storage.  If eigenvectors are
// desired, it uses a divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Zhpevd(jobz, uplo byte, n *int, ap *mat.CVector, w *mat.Vector, z *mat.CMatrix, ldz *int, work *mat.CVector, lwork *int, rwork *mat.Vector, lrwork *int, iwork *[]int, liwork, info *int) {
	var lquery, wantz bool
	var cone complex128
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var iinfo, imax, inde, indrwk, indtau, indwrk, iscale, liwmin, llrwk, llwrk, lrwmin, lwmin int

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters.
	wantz = jobz == 'V'
	lquery = ((*lwork) == -1 || (*lrwork) == -1 || (*liwork) == -1)

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

	if (*info) == 0 {
		if (*n) <= 1 {
			lwmin = 1
			liwmin = 1
			lrwmin = 1
		} else {
			if wantz {
				lwmin = 2 * (*n)
				lrwmin = 1 + 5*(*n) + 2*powint(*n, 2)
				liwmin = 3 + 5*(*n)
			} else {
				lwmin = (*n)
				lrwmin = (*n)
				liwmin = 1
			}
		}
		work.SetRe(0, float64(lwmin))
		rwork.Set(0, float64(lrwmin))
		(*iwork)[0] = liwmin

		if (*lwork) < lwmin && !lquery {
			(*info) = -9
		} else if (*lrwork) < lrwmin && !lquery {
			(*info) = -11
		} else if (*liwork) < liwmin && !lquery {
			(*info) = -13
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHPEVD"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if (*n) == 1 {
		w.Set(0, ap.GetRe(0))
		if wantz {
			z.Set(0, 0, cone)
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
	indrwk = inde + (*n)
	indwrk = indtau + (*n)
	llwrk = (*lwork) - indwrk + 1
	llrwk = (*lrwork) - indrwk + 1
	Zhptrd(uplo, n, ap, w, rwork.Off(inde-1), work.Off(indtau-1), &iinfo)

	//     For eigenvalues only, call DSTERF.  For eigenvectors, first call
	//     ZUPGTR to generate the orthogonal matrix, then call ZSTEDC.
	if !wantz {
		Dsterf(n, w, rwork.Off(inde-1), info)
	} else {
		Zstedc('I', n, w, rwork.Off(inde-1), z, ldz, work.Off(indwrk-1), &llwrk, rwork.Off(indrwk-1), &llrwk, iwork, liwork, info)
		Zupmtr('L', uplo, 'N', n, n, ap, work.Off(indtau-1), z, ldz, work.Off(indwrk-1), &iinfo)
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

	work.SetRe(0, float64(lwmin))
	rwork.Set(0, float64(lrwmin))
	(*iwork)[0] = liwmin
}
