package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhbevd computes all the eigenvalues and, optionally, eigenvectors of
// a complex Hermitian band matrix A.  If eigenvectors are desired, it
// uses a divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Zhbevd(jobz, uplo byte, n, kd *int, ab *mat.CMatrix, ldab *int, w *mat.Vector, z *mat.CMatrix, ldz *int, work *mat.CVector, lwork *int, rwork *mat.Vector, lrwork *int, iwork *[]int, liwork, info *int) {
	var lower, lquery, wantz bool
	var cone, czero complex128
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var iinfo, imax, inde, indwk2, indwrk, iscale, liwmin, llrwk, llwk2, lrwmin, lwmin int
	var err error
	_ = err

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters.
	wantz = jobz == 'V'
	lower = uplo == 'L'
	lquery = ((*lwork) == -1 || (*liwork) == -1 || (*lrwork) == -1)

	(*info) = 0
	if (*n) <= 1 {
		lwmin = 1
		lrwmin = 1
		liwmin = 1
	} else {
		if wantz {
			lwmin = 2 * powint(*n, 2)
			lrwmin = 1 + 5*(*n) + 2*powint(*n, 2)
			liwmin = 3 + 5*(*n)
		} else {
			lwmin = (*n)
			lrwmin = (*n)
			liwmin = 1
		}
	}
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

	if (*info) == 0 {
		work.SetRe(0, float64(lwmin))
		rwork.Set(0, float64(lrwmin))
		(*iwork)[0] = liwmin

		if (*lwork) < lwmin && !lquery {
			(*info) = -11
		} else if (*lrwork) < lrwmin && !lquery {
			(*info) = -13
		} else if (*liwork) < liwmin && !lquery {
			(*info) = -15
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHBEVD"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if (*n) == 1 {
		w.Set(0, ab.GetRe(0, 0))
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

	//     Call ZHBTRD to reduce Hermitian band matrix to tridiagonal form.
	inde = 1
	indwrk = inde + (*n)
	indwk2 = 1 + (*n)*(*n)
	llwk2 = (*lwork) - indwk2 + 1
	llrwk = (*lrwork) - indwrk + 1
	Zhbtrd(jobz, uplo, n, kd, ab, ldab, w, rwork.Off(inde-1), z, ldz, work, &iinfo)

	//     For eigenvalues only, call DSTERF.  For eigenvectors, call ZSTEDC.
	if !wantz {
		Dsterf(n, w, rwork.Off(inde-1), info)
	} else {
		Zstedc('I', n, w, rwork.Off(inde-1), work.CMatrix(*n, opts), n, work.Off(indwk2-1), &llwk2, rwork.Off(indwrk-1), &llrwk, iwork, liwork, info)
		err = goblas.Zgemm(NoTrans, NoTrans, *n, *n, *n, cone, z, *ldz, work.CMatrix(*n, opts), *n, czero, work.CMatrixOff(indwk2-1, *n, opts), *n)
		Zlacpy('A', n, n, work.CMatrixOff(indwk2-1, *n, opts), n, z, ldz)
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

	work.SetRe(0, float64(lwmin))
	rwork.Set(0, float64(lrwmin))
	(*iwork)[0] = liwmin
}
