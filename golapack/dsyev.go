package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsyev computes all eigenvalues and, optionally, eigenvectors of a
// real symmetric matrix A.
func Dsyev(jobz byte, uplo mat.MatUplo, n int, a *mat.Matrix, w, work *mat.Vector, lwork int) (info int, err error) {
	var lower, lquery, wantz bool
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var imax, inde, indtau, indwrk, iscale, llwork, lwkopt, nb int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'
	lower = uplo == Lower
	lquery = (lwork == -1)

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
		nb = Ilaenv(1, "Dsytrd", []byte{uplo.Byte()}, n, -1, -1, -1)
		lwkopt = max(1, (nb+2)*n)
		work.Set(0, float64(lwkopt))

		if lwork < max(1, 3*n-1) && !lquery {
			err = fmt.Errorf("lwork < max(1, 3*n-1) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dsyev", err)
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

	//     Call Dsytrd to reduce symmetric matrix to tridiagonal form.
	inde = 1
	indtau = inde + n
	indwrk = indtau + n
	llwork = lwork - indwrk + 1
	if err = Dsytrd(uplo, n, a, w, work.Off(inde-1), work.Off(indtau-1), work.Off(indwrk-1), llwork); err != nil {
		panic(err)
	}

	//     For eigenvalues only, call DSTERF.  For eigenvectors, first call
	//     DORGTR to generate the orthogonal matrix, then call DSTEQR.
	if !wantz {
		if info, err = Dsterf(n, w, work.Off(inde-1)); err != nil {
			panic(err)
		}
	} else {
		if err = Dorgtr(uplo, n, a, work.Off(indtau-1), work.Off(indwrk-1), llwork); err != nil {
			panic(err)
		}
		if info, err = Dsteqr(jobz, n, w, work.Off(inde-1), a, work.Off(indtau-1)); err != nil {
			panic(err)
		}
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
	if iscale == 1 {
		if info == 0 {
			imax = n
		} else {
			imax = info - 1
		}
		w.Scal(imax, one/sigma, 1)
	}

	//     Set WORK(1) to optimal workspace size.
	work.Set(0, float64(lwkopt))

	return
}
