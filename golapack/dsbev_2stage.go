package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsbev2stage computes all the eigenvalues and, optionally, eigenvectors of
// a real symmetric band matrix A using the 2stage technique for
// the reduction to tridiagonal.
func Dsbev2stage(jobz byte, uplo mat.MatUplo, n, kd int, ab *mat.Matrix, w *mat.Vector, z *mat.Matrix, work *mat.Vector, lwork int) (info int, err error) {
	var lower, lquery, wantz bool
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var ib, imax, inde, indhous, indwrk, iscale, lhtrd, llwork, lwmin, lwtrd int

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
	} else if kd < 0 {
		err = fmt.Errorf("kd < 0: kd=%v", kd)
	} else if ab.Rows < kd+1 {
		err = fmt.Errorf("ab.Rows < kd+1: ab.Rows=%v, kd=%v", ab.Rows, kd)
	} else if z.Rows < 1 || (wantz && z.Rows < n) {
		err = fmt.Errorf("z.Rows < 1 || (wantz && z.Rows < n): jobz='%c', z.Rows=%v, n=%v", jobz, z.Rows, n)
	}

	if err == nil {
		if n <= 1 {
			lwmin = 1
			work.Set(0, float64(lwmin))
		} else {
			ib = Ilaenv2stage(2, "DsytrdSb2st", []byte{jobz}, n, kd, -1, -1)
			lhtrd = Ilaenv2stage(3, "DsytrdSb2st", []byte{jobz}, n, kd, ib, -1)
			lwtrd = Ilaenv2stage(4, "DsytrdSb2st", []byte{jobz}, n, kd, ib, -1)
			lwmin = n + lhtrd + lwtrd
			work.Set(0, float64(lwmin))
		}

		if lwork < lwmin && !lquery {
			err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dsbev2stage", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if n == 1 {
		if lower {
			w.Set(0, ab.Get(0, 0))
		} else {
			w.Set(0, ab.Get(kd, 0))
		}
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
	anrm = Dlansb('M', uplo, n, kd, ab, work)
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
			if err = Dlascl('B', kd, kd, one, sigma, n, n, ab); err != nil {
				panic(err)
			}
		} else {
			if err = Dlascl('Q', kd, kd, one, sigma, n, n, ab); err != nil {
				panic(err)
			}
		}
	}

	//     Call DsytrdSb2st to reduce symmetric band matrix to tridiagonal form.
	inde = 1
	indhous = inde + n
	indwrk = indhous + lhtrd
	llwork = lwork - indwrk + 1

	if err = DsytrdSb2st('N', jobz, uplo, n, kd, ab, w, work.Off(inde-1), work.Off(indhous-1), lhtrd, work.Off(indwrk-1), llwork); err != nil {
		panic(err)
	}

	//     For eigenvalues only, call DSTERF.  For eigenvectors, call SSTEQR.
	if !wantz {
		if info, err = Dsterf(n, w, work.Off(inde-1)); err != nil {
			panic(err)
		}
	} else {
		if info, err = Dsteqr(jobz, n, w, work.Off(inde-1), z, work.Off(indwrk-1)); err != nil {
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
	work.Set(0, float64(lwmin))

	return
}
