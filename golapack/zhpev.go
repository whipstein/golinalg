package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhpev computes all the eigenvalues and, optionally, eigenvectors of a
// complex Hermitian matrix in packed storage.
func Zhpev(jobz byte, uplo mat.MatUplo, n int, ap *mat.CVector, w *mat.Vector, z *mat.CMatrix, work *mat.CVector, rwork *mat.Vector) (info int, err error) {
	var wantz bool
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var imax, inde, indrwk, indtau, indwrk, iscale int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'

	if !(wantz || jobz == 'N') {
		err = fmt.Errorf("!(wantz || jobz == 'N'): jobz='%c'", jobz)
	} else if !(uplo == Lower || uplo == Upper) {
		err = fmt.Errorf("!(uplo == Lower || uplo == Upper): uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if z.Rows < 1 || (wantz && z.Rows < n) {
		err = fmt.Errorf("z.Rows < 1 || (wantz && z.Rows < n): jobz='%c', z.Rows=%v, n=%v", jobz, z.Rows, n)
	}

	if err != nil {
		gltest.Xerbla2("Zhpev", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if n == 1 {
		w.Set(0, ap.GetRe(0))
		rwork.Set(0, 1)
		if wantz {
			z.SetRe(0, 0, one)
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
		goblas.Zdscal((n*(n+1))/2, sigma, ap.Off(0, 1))
	}

	//     Call ZHPTRD to reduce Hermitian packed matrix to tridiagonal form.
	inde = 1
	indtau = 1
	if err = Zhptrd(uplo, n, ap, w, rwork.Off(inde-1), work.Off(indtau-1)); err != nil {
		panic(err)
	}

	//     For eigenvalues only, call DSTERF.  For eigenvectors, first call
	//     ZUPGTR to generate the orthogonal matrix, then call ZSTEQR.
	if !wantz {
		if info, err = Dsterf(n, w, rwork.Off(inde-1)); err != nil {
			panic(err)
		}
	} else {
		indwrk = indtau + n
		if err = Zupgtr(uplo, n, ap, work.Off(indtau-1), z, work.Off(indwrk-1)); err != nil {
			panic(err)
		}
		indrwk = inde + n
		if info, err = Zsteqr(jobz, n, w, rwork.Off(inde-1), z, rwork.Off(indrwk-1)); err != nil {
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
