package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsyev2stage computes all eigenvalues and, optionally, eigenvectors of a
// real symmetric matrix A using the 2stage technique for
// the reduction to tridiagonal.
func Dsyev2stage(jobz byte, uplo mat.MatUplo, n int, a *mat.Matrix, w, work *mat.Vector, lwork int) (info int, err error) {
	var lower, lquery, wantz bool
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var ib, imax, inde, indhous, indtau, indwrk, iscale, kd, lhtrd, llwork, lwmin, lwtrd int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'
	lower = uplo == Lower
	lquery = (lwork == -1)

	if jobz != 'N' {
		err = fmt.Errorf("jobz != 'N': jobz='%c'", jobz)
	} else if !(lower || uplo == Upper) {
		err = fmt.Errorf("!(lower || uplo == Upper): uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}

	if err == nil {
		kd = Ilaenv2stage(1, "Dsytrd2stage", []byte{jobz}, n, -1, -1, -1)
		ib = Ilaenv2stage(2, "Dsytrd2stage", []byte{jobz}, n, kd, -1, -1)
		lhtrd = Ilaenv2stage(3, "Dsytrd2stage", []byte{jobz}, n, kd, ib, -1)
		lwtrd = Ilaenv2stage(4, "Dsytrd2stage", []byte{jobz}, n, kd, ib, -1)
		lwmin = 2*n + lhtrd + lwtrd
		work.Set(0, float64(lwmin))

		if lwork < lwmin && !lquery {
			err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dsyev2stage", err)
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
		work.Set(0, 2)
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

	//     Call Dsytrd2stage to reduce symmetric matrix to tridiagonal form.
	inde = 1
	indtau = inde + n
	indhous = indtau + n
	indwrk = indhous + lhtrd
	llwork = lwork - indwrk + 1

	if err = Dsytrd2stage(jobz, uplo, n, a, w, work.Off(inde-1), work.Off(indtau-1), work.Off(indhous-1), lhtrd, work.Off(indwrk-1), llwork); err != nil {
		panic(err)
	}

	//     For eigenvalues only, call DSTERF.  For eigenvectors, first call
	//     DORGTR to generate the orthogonal matrix, then call DSTEQR.
	if !wantz {
		if info, err = Dsterf(n, w, work.Off(inde-1)); err != nil {
			panic(err)
		}
	} else {
		//        Not available in this release, and argument checking should not
		//        let it getting here
		return
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
	if iscale == 1 {
		if info == 0 {
			imax = n
		} else {
			imax = info - 1
		}
		goblas.Dscal(imax, one/sigma, w.Off(0, 1))
	}

	//     Set WORK(1) to optimal workspace size.
	work.Set(0, float64(lwmin))

	return
}
