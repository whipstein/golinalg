package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zeevd2stage computes all eigenvalues and, optionally, eigenvectors of a
// complex Hermitian matrix A using the 2stage technique for
// the reduction to tridiagonal.  If eigenvectors are desired, it uses a
// divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Zheevd2stage(jobz byte, uplo mat.MatUplo, n int, a *mat.CMatrix, w *mat.Vector, work *mat.CVector, lwork int, rwork *mat.Vector, lrwork int, iwork *[]int, liwork int) (info int, err error) {
	var lower, lquery, wantz bool
	var cone complex128
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var ib, imax, inde, indhous, indrwk, indtau, indwk2, indwrk, iscale, kd, lhtrd, liwmin, llrwk, llwork, llwrk2, lrwmin, lwmin, lwtrd int

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters.
	wantz = jobz == 'V'
	lower = uplo == Lower
	lquery = (lwork == -1 || lrwork == -1 || liwork == -1)

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
		if n <= 1 {
			lwmin = 1
			lrwmin = 1
			liwmin = 1
		} else {
			kd = Ilaenv2stage(1, "Zhetrd2stage", []byte{jobz}, n, -1, -1, -1)
			ib = Ilaenv2stage(2, "Zhetrd2stage", []byte{jobz}, n, kd, -1, -1)
			lhtrd = Ilaenv2stage(3, "Zhetrd2stage", []byte{jobz}, n, kd, ib, -1)
			lwtrd = Ilaenv2stage(4, "Zhetrd2stage", []byte{jobz}, n, kd, ib, -1)
			if wantz {
				lwmin = 2*n + n*n
				lrwmin = 1 + 5*n + 2*pow(n, 2)
				liwmin = 3 + 5*n
			} else {
				lwmin = n + 1 + lhtrd + lwtrd
				lrwmin = n
				liwmin = 1
			}
		}
		work.SetRe(0, float64(lwmin))
		rwork.Set(0, float64(lrwmin))
		(*iwork)[0] = liwmin

		if lwork < lwmin && !lquery {
			err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
		} else if lrwork < lrwmin && !lquery {
			err = fmt.Errorf("lrwork < lrwmin && !lquery: lrwork=%v, lrwmin=%v, lquery=%v", lrwork, lrwmin, lquery)
		} else if liwork < liwmin && !lquery {
			err = fmt.Errorf("liwork < liwmin && !lquery: liwork=%v, liwmin=%v, lquery=%v", liwork, liwmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zheevd2stage", err)
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

	//     Call ZHETRD_2STAGE to reduce Hermitian matrix to tridiagonal form.
	inde = 1
	indrwk = inde + n
	llrwk = lrwork - indrwk + 1
	indtau = 1
	indhous = indtau + n
	indwrk = indhous + lhtrd
	llwork = lwork - indwrk + 1
	indwk2 = indwrk + n*n
	llwrk2 = lwork - indwk2 + 1

	if err = Zhetrd2stage(jobz, uplo, n, a, w, rwork.Off(inde-1), work.Off(indtau-1), work.Off(indhous-1), lhtrd, work.Off(indwrk-1), llwork); err != nil {
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
		if info, err = Zstedc('I', n, w, rwork.Off(inde-1), work.CMatrixOff(indwrk-1, n, opts), work.Off(indwk2-1), llwrk2, rwork.Off(indrwk-1), llrwk, iwork, liwork); err != nil {
			panic(err)
		}
		if err = Zunmtr(Left, uplo, NoTrans, n, n, a, work.Off(indtau-1), work.CMatrixOff(indwrk-1, n, opts), work.Off(indwk2-1), llwrk2); err != nil {
			panic(err)
		}
		Zlacpy(Full, n, n, work.CMatrixOff(indwrk-1, n, opts), a)
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

	work.SetRe(0, float64(lwmin))
	rwork.Set(0, float64(lrwmin))
	(*iwork)[0] = liwmin

	return
}
