package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zheevd computes all eigenvalues and, optionally, eigenvectors of a
// complex Hermitian matrix A.  If eigenvectors are desired, it uses a
// divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Zheevd(jobz byte, uplo mat.MatUplo, n int, a *mat.CMatrix, w *mat.Vector, work *mat.CVector, lwork int, rwork *mat.Vector, lrwork int, iwork *[]int, liwork int) (info int, err error) {
	var lower, lquery, wantz bool
	var cone complex128
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var imax, inde, indrwk, indtau, indwk2, indwrk, iscale, liopt, liwmin, llrwk, llwork, llwrk2, lopt, lropt, lrwmin, lwmin int

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters.
	wantz = jobz == 'V'
	lower = uplo == Lower
	lquery = (lwork == -1 || lrwork == -1 || liwork == -1)

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
			lwmin = 1
			lrwmin = 1
			liwmin = 1
			lopt = lwmin
			lropt = lrwmin
			liopt = liwmin
		} else {
			if wantz {
				lwmin = 2*n + n*n
				lrwmin = 1 + 5*n + 2*pow(n, 2)
				liwmin = 3 + 5*n
			} else {
				lwmin = n + 1
				lrwmin = n
				liwmin = 1
			}
			lopt = max(lwmin, n+Ilaenv(1, "Zhetrd", []byte{uplo.Byte()}, n, -1, -1, -1))
			lropt = lrwmin
			liopt = liwmin
		}
		work.SetRe(0, float64(lopt))
		rwork.Set(0, float64(lropt))
		(*iwork)[0] = liopt

		if lwork < lwmin && !lquery {
			err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
		} else if lrwork < lrwmin && !lquery {
			err = fmt.Errorf("lrwork < lrwmin && !lquery: lrwork=%v, lrwmin=%v, lquery=%v", lrwork, lrwmin, lquery)
		} else if liwork < liwmin && !lquery {
			err = fmt.Errorf("liwork < liwmin && !lquery: liwork=%v, liwmin=%v, lquery=%v", liwork, liwmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zheevd", err)
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
	indrwk = inde + n
	indwk2 = indwrk + n*n
	llwork = lwork - indwrk + 1
	llwrk2 = lwork - indwk2 + 1
	llrwk = lrwork - indrwk + 1
	if err = Zhetrd(uplo, n, a, w, rwork.Off(inde-1), work.Off(indtau-1), work.Off(indwrk-1), llwork); err != nil {
		panic(err)
	}

	//     For eigenvalues only, call DSTERF.  For eigenvectors, first call
	//     ZSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
	//     tridiagonal matrix, then call ZUNMTR to multiply it to the
	//     Householder transformations represented as Householder vectors in
	//     A.
	if !wantz {
		if info, err = Dsterf(n, w, rwork.Off(inde-1)); err != nil {
			panic(err)
		}
	} else {
		if info, err = Zstedc('I', n, w, rwork.Off(inde-1), work.Off(indwrk-1).CMatrix(n, opts), work.Off(indwk2-1), llwrk2, rwork.Off(indrwk-1), llrwk, iwork, liwork); err != nil {
			panic(err)
		}
		if err = Zunmtr(Left, uplo, NoTrans, n, n, a, work.Off(indtau-1), work.Off(indwrk-1).CMatrix(n, opts), work.Off(indwk2-1), llwrk2); err != nil {
			panic(err)
		}
		Zlacpy(Full, n, n, work.Off(indwrk-1).CMatrix(n, opts), a)
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

	work.SetRe(0, float64(lopt))
	rwork.Set(0, float64(lropt))
	(*iwork)[0] = liopt

	return
}
