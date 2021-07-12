package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsyevd computes all eigenvalues and, optionally, eigenvectors of a
// real symmetric matrix A. If eigenvectors are desired, it uses a
// divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
//
// Because of large use of BLAS of level 3, DSYEVD needs N**2 more
// workspace than DSYEVX.
func Dsyevd(jobz, uplo byte, n *int, a *mat.Matrix, lda *int, w, work *mat.Vector, lwork *int, iwork *[]int, liwork, info *int) {
	var lower, lquery, wantz bool
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var iinfo, inde, indtau, indwk2, indwrk, iscale, liopt, liwmin, llwork, llwrk2, lopt, lwmin int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'
	lower = uplo == 'L'
	lquery = ((*lwork) == -1 || (*liwork) == -1)

	(*info) = 0
	if !(wantz || jobz == 'N') {
		(*info) = -1
	} else if !(lower || uplo == 'U') {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*lda) < max(1, *n) {
		(*info) = -5
	}

	if (*info) == 0 {
		if (*n) <= 1 {
			liwmin = 1
			lwmin = 1
			lopt = lwmin
			liopt = liwmin
		} else {
			if wantz {
				liwmin = 3 + 5*(*n)
				lwmin = 1 + 6*(*n) + 2*int(math.Pow(float64(*n), 2))
			} else {
				liwmin = 1
				lwmin = 2*(*n) + 1
			}
			lopt = max(lwmin, 2*(*n)+Ilaenv(toPtr(1), []byte("DSYTRD"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1)))
			liopt = liwmin
		}
		work.Set(0, float64(lopt))
		(*iwork)[0] = liopt

		if (*lwork) < lwmin && !lquery {
			(*info) = -8
		} else if (*liwork) < liwmin && !lquery {
			(*info) = -10
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYEVD"), -(*info))
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

	//     Call DSYTRD to reduce symmetric matrix to tridiagonal form.
	inde = 1
	indtau = inde + (*n)
	indwrk = indtau + (*n)
	llwork = (*lwork) - indwrk + 1
	indwk2 = indwrk + (*n)*(*n)
	llwrk2 = (*lwork) - indwk2 + 1

	Dsytrd(uplo, n, a, lda, w, work.Off(inde-1), work.Off(indtau-1), work.Off(indwrk-1), &llwork, &iinfo)

	//     For eigenvalues only, call DSTERF.  For eigenvectors, first call
	//     DSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
	//     tridiagonal matrix, then call DORMTR to multiply it by the
	//     Householder transformations stored in A.
	if !wantz {
		Dsterf(n, w, work.Off(inde-1), info)
	} else {
		Dstedc('I', n, w, work.Off(inde-1), work.MatrixOff(indwrk-1, *n, opts), n, work.Off(indwk2-1), &llwrk2, iwork, liwork, info)
		Dormtr('L', uplo, 'N', n, n, a, lda, work.Off(indtau-1), work.MatrixOff(indwrk-1, *n, opts), n, work.Off(indwk2-1), &llwrk2, &iinfo)
		Dlacpy('A', n, n, work.MatrixOff(indwrk-1, *n, opts), n, a, lda)
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
	if iscale == 1 {
		goblas.Dscal(*n, one/sigma, w.Off(0, 1))
	}

	work.Set(0, float64(lopt))
	(*iwork)[0] = liopt
}
