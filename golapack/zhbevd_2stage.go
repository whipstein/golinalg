package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhbevd2stage computes all the eigenvalues and, optionally, eigenvectors of
// a complex Hermitian band matrix A using the 2stage technique for
// the reduction to tridiagonal.  If eigenvectors are desired, it
// uses a divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Zhbevd2stage(jobz byte, uplo mat.MatUplo, n, kd int, ab *mat.CMatrix, w *mat.Vector, z *mat.CMatrix, work *mat.CVector, lwork int, rwork *mat.Vector, lrwork int, iwork *[]int, liwork int) (info int, err error) {
	var lower, lquery, wantz bool
	var cone, czero complex128
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var ib, imax, inde, indhous, indrwk, indwk, indwk2, iscale, lhtrd, liwmin, llrwk, llwk2, llwork, lrwmin, lwmin, lwtrd int

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters.
	wantz = jobz == 'V'
	lower = uplo == Lower
	lquery = (lwork == -1 || liwork == -1 || lrwork == -1)

	if n <= 1 {
		lwmin = 1
		lrwmin = 1
		liwmin = 1
	} else {
		ib = Ilaenv2stage(2, "ZhetrdHb2st", []byte{jobz}, n, kd, -1, -1)
		lhtrd = Ilaenv2stage(3, "ZhetrdHb2st", []byte{jobz}, n, kd, ib, -1)
		lwtrd = Ilaenv2stage(4, "ZhetrdHb2st", []byte{jobz}, n, kd, ib, -1)
		if wantz {
			lwmin = 2 * pow(n, 2)
			lrwmin = 1 + 5*n + 2*pow(n, 2)
			liwmin = 3 + 5*n
		} else {
			lwmin = max(n, lhtrd+lwtrd)
			lrwmin = n
			liwmin = 1
		}
	}
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
		gltest.Xerbla2("Zhbevd2stage", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if n == 1 {
		w.Set(0, ab.GetRe(0, 0))
		if wantz {
			z.Set(0, 0, cone)
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

	//     Call ZHBTRD_HB2ST to reduce Hermitian band matrix to tridiagonal form.
	inde = 1
	indrwk = inde + n
	llrwk = lrwork - indrwk + 1
	indhous = 1
	indwk = indhous + lhtrd
	llwork = lwork - indwk + 1
	indwk2 = indwk + n*n
	llwk2 = lwork - indwk2 + 1

	if err = ZhetrdHb2st('N', jobz, uplo, n, kd, ab, w, rwork.Off(inde-1), work.Off(indhous-1), lhtrd, work.Off(indwk-1), llwork); err != nil {
		panic(err)
	}

	//     For eigenvalues only, call DSTERF.  For eigenvectors, call ZSTEDC.
	if !wantz {
		if info, err = Dsterf(n, w, rwork.Off(inde-1)); err != nil {
			panic(err)
		}
	} else {
		if info, err = Zstedc('I', n, w, rwork.Off(inde-1), work.CMatrix(n, opts), work.Off(indwk2-1), llwk2, rwork.Off(indrwk-1), llrwk, iwork, liwork); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, NoTrans, n, n, n, cone, z, work.CMatrix(n, opts), czero, work.CMatrixOff(indwk2-1, n, opts)); err != nil {
			panic(err)
		}
		Zlacpy(Full, n, n, work.CMatrixOff(indwk2-1, n, opts), z)
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
