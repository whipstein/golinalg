package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zheev computes all eigenvalues and, optionally, eigenvectors of a
// complex Hermitian matrix A.
func Zheev(jobz byte, uplo mat.MatUplo, n int, a *mat.CMatrix, w *mat.Vector, work *mat.CVector, lwork int, rwork *mat.Vector) (info int, err error) {
	var lower, lquery, wantz bool
	var cone complex128
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var imax, inde, indtau, indwrk, iscale, llwork, lwkopt, nb int

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)

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
		nb = Ilaenv(1, "Zhetrd", []byte{uplo.Byte()}, n, -1, -1, -1)
		lwkopt = max(1, (nb+1)*n)
		work.SetRe(0, float64(lwkopt))

		if lwork < max(1, 2*n-1) && !lquery {
			err = fmt.Errorf("lwork < max(1, 2*n-1) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zheev", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if n == 1 {
		w.Set(0, a.GetRe(0, 0))
		work.Set(0, 1)
		if wantz {
			a.Set(0, 0, cone)
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
	anrm = Zlanhe('M', uplo, n, a, rwork)
	iscale = 0
	if anrm > zero && anrm < rmin {
		iscale = 1
		sigma = rmin / anrm
	} else if anrm > rmax {
		iscale = 1
		sigma = rmax / anrm
	}
	if iscale == 1 {
		if err = Zlascl(uplo.Byte(), 0, 0, one, sigma, n, n, a); err != nil {
			panic(err)
		}
	}

	//     Call Zhetrd to reduce Hermitian matrix to tridiagonal form.
	inde = 1
	indtau = 1
	indwrk = indtau + n
	llwork = lwork - indwrk + 1
	if err = Zhetrd(uplo, n, a, w, rwork.Off(inde-1), work.Off(indtau-1), work.Off(indwrk-1), llwork); err != nil {
		panic(err)
	}

	//     For eigenvalues only, call DSTERF.  For eigenvectors, first call
	//     ZUNGTR to generate the unitary matrix, then call ZSTEQR.
	if !wantz {
		if info, err = Dsterf(n, w, rwork.Off(inde-1)); err != nil {
			panic(err)
		}
	} else {
		if err = Zungtr(uplo, n, a, work.Off(indtau-1), work.Off(indwrk-1), llwork); err != nil {
			panic(err)
		}
		indwrk = inde + n
		if info, err = Zsteqr(jobz, n, w, rwork.Off(inde-1), a, rwork.Off(indwrk-1)); err != nil {
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

	//     Set WORK(1) to optimal complex workspace size.
	work.SetRe(0, float64(lwkopt))

	return
}
