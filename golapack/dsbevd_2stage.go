package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsbevd2stage computes all the eigenvalues and, optionally, eigenvectors of
// a real symmetric band matrix A using the 2stage technique for
// the reduction to tridiagonal. If eigenvectors are desired, it uses
// a divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Dsbevd2stage(jobz, uplo byte, n, kd *int, ab *mat.Matrix, ldab *int, w *mat.Vector, z *mat.Matrix, ldz *int, work *mat.Vector, lwork *int, iwork *[]int, liwork, info *int) {
	var lower, lquery, wantz bool
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var ib, iinfo, inde, indhous, indwk2, indwrk, iscale, lhtrd, liwmin, llwork, llwrk2, lwmin, lwtrd int
	var err error
	_ = err

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'
	lower = uplo == 'L'
	lquery = ((*lwork) == -1 || (*liwork) == -1)

	(*info) = 0
	if (*n) <= 1 {
		liwmin = 1
		lwmin = 1
	} else {
		ib = Ilaenv2stage(toPtr(2), []byte("DSYTRD_SB2ST"), []byte{jobz}, n, kd, toPtr(-1), toPtr(-1))
		lhtrd = Ilaenv2stage(toPtr(3), []byte("DSYTRD_SB2ST"), []byte{jobz}, n, kd, &ib, toPtr(-1))
		lwtrd = Ilaenv2stage(toPtr(4), []byte("DSYTRD_SB2ST"), []byte{jobz}, n, kd, &ib, toPtr(-1))
		if wantz {
			liwmin = 3 + 5*(*n)
			lwmin = 1 + 5*(*n) + 2*int(math.Pow(float64(*n), 2))
		} else {
			liwmin = 1
			lwmin = maxint(2*(*n), (*n)+lhtrd+lwtrd)
		}
	}
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
		work.Set(0, float64(lwmin))
		(*iwork)[0] = liwmin

		if (*lwork) < lwmin && !lquery {
			(*info) = -11
		} else if (*liwork) < liwmin && !lquery {
			(*info) = -13
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSBEVD_2STAGE"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if (*n) == 1 {
		w.Set(0, ab.Get(0, 0))
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

	//     Call DSYTRD_SB2ST to reduce band symmetric matrix to tridiagonal form.
	inde = 1
	indhous = inde + (*n)
	indwrk = indhous + lhtrd
	llwork = (*lwork) - indwrk + 1
	indwk2 = indwrk + (*n)*(*n)
	llwrk2 = (*lwork) - indwk2 + 1

	DsytrdSb2st('N', jobz, uplo, n, kd, ab, ldab, w, work.Off(inde-1), work.Off(indhous-1), &lhtrd, work.Off(indwrk-1), &llwork, &iinfo)

	//     For eigenvalues only, call DSTERF.  For eigenvectors, call SSTEDC.
	if !wantz {
		Dsterf(n, w, work.Off(inde-1), info)
	} else {
		Dstedc('I', n, w, work.Off(inde-1), work.MatrixOff(indwrk-1, *n, opts), n, work.Off(indwk2-1), &llwrk2, iwork, liwork, info)
		err = goblas.Dgemm(NoTrans, NoTrans, *n, *n, *n, one, z, *ldz, work.MatrixOff(indwrk-1, *n, opts), *n, zero, work.MatrixOff(indwk2-1, *n, opts), *n)
		Dlacpy('A', n, n, work.MatrixOff(indwk2-1, *n, opts), n, z, ldz)
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
	if iscale == 1 {
		goblas.Dscal(*n, one/sigma, w, 1)
	}

	work.Set(0, float64(lwmin))
	(*iwork)[0] = liwmin
}
