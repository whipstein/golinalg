package golapack

import (
	"fmt"
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
// Because of large use of BLAS of level 3, Dsyevd needs N**2 more
// workspace than DSYEVX.
func Dsyevd(jobz byte, uplo mat.MatUplo, n int, a *mat.Matrix, w, work *mat.Vector, lwork int, iwork *[]int, liwork int) (info int, err error) {
	var lower, lquery, wantz bool
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var inde, indtau, indwk2, indwrk, iscale, liopt, liwmin, llwork, llwrk2, lopt, lwmin int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'
	lower = uplo == Lower
	lquery = (lwork == -1 || liwork == -1)

	if !(wantz || jobz == 'N') {
		err = fmt.Errorf("!(wantz || jobz == 'N'): jobz='%c'", jobz)
	} else if !(lower || uplo == Upper) {
		err = fmt.Errorf("!(lower || uplo == Upper): uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}

	if err == nil {
		if n <= 1 {
			liwmin = 1
			lwmin = 1
			lopt = lwmin
			liopt = liwmin
		} else {
			if wantz {
				liwmin = 3 + 5*n
				lwmin = 1 + 6*n + 2*pow(n, 2)
			} else {
				liwmin = 1
				lwmin = 2*n + 1
			}
			lopt = max(lwmin, 2*n+Ilaenv(1, "Dsytrd", []byte{uplo.Byte()}, n, -1, -1, -1))
			liopt = liwmin
		}
		work.Set(0, float64(lopt))
		(*iwork)[0] = liopt

		if lwork < lwmin && !lquery {
			err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
		} else if liwork < liwmin && !lquery {
			err = fmt.Errorf("liwork < liwmin && !lquery: liwork=%v, liwmin=%v, lquery=%v", liwork, liwmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dsyevd", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if n == 1 {
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
	anrm = Dlansy('M', uplo, n, a, work)
	iscale = 0
	if anrm > zero && anrm < rmin {
		iscale = 1
		sigma = rmin / anrm
	} else if anrm > rmax {
		iscale = 1
		sigma = rmax / anrm
	}
	if iscale == 1 {
		if err = Dlascl(uplo.Byte(), 0, 0, one, sigma, n, n, a); err != nil {
			panic(err)
		}
	}

	//     Call Dsytrd to reduce symmetric matrix to tridiagonal form.
	inde = 1
	indtau = inde + n
	indwrk = indtau + n
	llwork = lwork - indwrk + 1
	indwk2 = indwrk + n*n
	llwrk2 = lwork - indwk2 + 1

	if err = Dsytrd(uplo, n, a, w, work.Off(inde-1), work.Off(indtau-1), work.Off(indwrk-1), llwork); err != nil {
		panic(err)
	}

	//     For eigenvalues only, call DSTERF.  For eigenvectors, first call
	//     DSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
	//     tridiagonal matrix, then call DORMTR to multiply it by the
	//     Householder transformations stored in A.
	if !wantz {
		if info, err = Dsterf(n, w, work.Off(inde-1)); err != nil {
			panic(err)
		}
	} else {
		if info, err = Dstedc('I', n, w, work.Off(inde-1), work.MatrixOff(indwrk-1, n, opts), work.Off(indwk2-1), llwrk2, iwork, liwork); err != nil {
			panic(err)
		}
		if err = Dormtr(Left, uplo, NoTrans, n, n, a, work.Off(indtau-1), work.MatrixOff(indwrk-1, n, opts), work.Off(indwk2-1), llwrk2); err != nil {
			panic(err)
		}
		Dlacpy(Full, n, n, work.MatrixOff(indwrk-1, n, opts), a)
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
	if iscale == 1 {
		goblas.Dscal(n, one/sigma, w.Off(0, 1))
	}

	work.Set(0, float64(lopt))
	(*iwork)[0] = liopt

	return
}
