package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dspev computes all the eigenvalues and, optionally, eigenvectors of a
// real symmetric matrix A in packed storage.
func Dspev(jobz byte, uplo mat.MatUplo, n int, ap, w *mat.Vector, z *mat.Matrix, work *mat.Vector) (info int, err error) {
	var wantz bool
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var imax, inde, indtau, indwrk, iscale int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'

	if !(wantz || jobz == 'N') {
		err = fmt.Errorf("!(wantz || jobz == 'N'): jobz='%c'", jobz)
	} else if !(uplo == Upper || uplo == Lower) {
		err = fmt.Errorf("!(uplo == Upper || uplo == Lower): uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if z.Rows < 1 || (wantz && z.Rows < n) {
		err = fmt.Errorf("z.Rows < 1 || (wantz && z.Rows < n): jobz='%c', z.Rows=%v, n=%v", jobz, z.Rows, n)
	}

	if err != nil {
		gltest.Xerbla2("Dspev", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if n == 1 {
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
		goblas.Dscal((n*(n+1))/2, sigma, ap.Off(0, 1))
	}

	//     Call DSPTRD to reduce symmetric packed matrix to tridiagonal form.
	inde = 1
	indtau = inde + n
	if err = Dsptrd(uplo, n, ap, w, work.Off(inde-1), work.Off(indtau-1)); err != nil {
		panic(err)
	}

	//     For eigenvalues only, call DSTERF.  For eigenvectors, first call
	//     DOPGTR to generate the orthogonal matrix, then call DSTEQR.
	if !wantz {
		if info, err = Dsterf(n, w, work.Off(inde-1)); err != nil {
			panic(err)
		}
	} else {
		indwrk = indtau + n
		if err = Dopgtr(uplo, n, ap, work.Off(indtau-1), z, work.Off(indwrk-1)); err != nil {
			panic(err)
		}
		if info, err = Dsteqr(jobz, n, w, work.Off(inde-1), z, work.Off(indtau-1)); err != nil {
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
		goblas.Dscal(imax, one/sigma, w.Off(0, 1))
	}

	return
}
