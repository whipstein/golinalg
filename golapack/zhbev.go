package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhbev computes all the eigenvalues and, optionally, eigenvectors of
// a complex Hermitian band matrix A.
func Zhbev(jobz byte, uplo mat.MatUplo, n, kd int, ab *mat.CMatrix, w *mat.Vector, z *mat.CMatrix, work *mat.CVector, rwork *mat.Vector) (info int, err error) {
	var lower, wantz bool
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var imax, inde, indrwk, iscale int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'
	lower = uplo == Lower

	if !(wantz || jobz == 'N') {
		err = fmt.Errorf("!(wantz || jobz == 'N'): jobz='%c'", jobz)
	} else if !(lower || uplo == Upper) {
		err = fmt.Errorf("!(lower || uplo == Upper): uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if kd < 0 {
		err = fmt.Errorf("kd < 0: kd=%v", kd)
	} else if ab.Rows < kd+1 {
		err = fmt.Errorf("ab.Rows < kd+1: ab.Rows=%v, kd=%v", ab.Rows, kd)
	} else if z.Rows < 1 || (wantz && z.Rows < n) {
		err = fmt.Errorf("z.Rows < 1 || (wantz && z.Rows < n): jobz='%c', z.Rows=%v, n=%v", jobz, z.Rows, n)
	}

	if err != nil {
		gltest.Xerbla2("Zhbev", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if n == 1 {
		if lower {
			w.Set(0, ab.GetRe(0, 0))
		} else {
			w.Set(0, ab.GetRe(kd, 0))
		}
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
	anrm = Zlanhb('M', uplo, n, kd, ab, rwork)
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
			if err = Zlascl('B', kd, kd, one, sigma, n, n, ab); err != nil {
				panic(err)
			}
		} else {
			if err = Zlascl('Q', kd, kd, one, sigma, n, n, ab); err != nil {
				panic(err)
			}
		}
	}

	//     Call ZHBTRD to reduce Hermitian band matrix to tridiagonal form.
	inde = 1
	if err = Zhbtrd(jobz, uplo, n, kd, ab, w, rwork.Off(inde-1), z, work); err != nil {
		panic(err)
	}

	//     For eigenvalues only, call DSTERF.  For eigenvectors, call ZSTEQR.
	if !wantz {
		if info, err = Dsterf(n, w, rwork.Off(inde-1)); err != nil {
			panic(err)
		}
	} else {
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
