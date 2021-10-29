package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsbevd2stage computes all the eigenvalues and, optionally, eigenvectors of
// a real symmetric band matrix A using the 2stage technique for
// the reduction to tridiagonal. If eigenvectors are desired, it uses
// a divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Dsbevd2stage(jobz byte, uplo mat.MatUplo, n, kd int, ab *mat.Matrix, w *mat.Vector, z *mat.Matrix, work *mat.Vector, lwork int, iwork *[]int, liwork int) (info int, err error) {
	var lower, lquery, wantz bool
	var anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, zero float64
	var ib, inde, indhous, indwk2, indwrk, iscale, lhtrd, liwmin, llwork, llwrk2, lwmin, lwtrd int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'
	lower = uplo == Lower
	lquery = (lwork == -1 || liwork == -1)

	if n <= 1 {
		liwmin = 1
		lwmin = 1
	} else {
		ib = Ilaenv2stage(2, "DsytrdSb2st", []byte{jobz}, n, kd, -1, -1)
		lhtrd = Ilaenv2stage(3, "DsytrdSb2st", []byte{jobz}, n, kd, ib, -1)
		lwtrd = Ilaenv2stage(4, "DsytrdSb2st", []byte{jobz}, n, kd, ib, -1)
		if wantz {
			liwmin = 3 + 5*n
			lwmin = 1 + 5*n + 2*pow(n, 2)
		} else {
			liwmin = 1
			lwmin = max(2*n, n+lhtrd+lwtrd)
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
		work.Set(0, float64(lwmin))
		(*iwork)[0] = liwmin

		if lwork < lwmin && !lquery {
			err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
		} else if liwork < liwmin && !lquery {
			err = fmt.Errorf("liwork < liwmin && !lquery: liwork=%v, liwmin=%v, lquery=%v", liwork, liwmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dsbevd2stage", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if n == 1 {
		w.Set(0, ab.Get(0, 0))
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

	//     Call DsytrdSb2st to reduce band symmetric matrix to tridiagonal form.
	inde = 1
	indhous = inde + n
	indwrk = indhous + lhtrd
	llwork = lwork - indwrk + 1
	indwk2 = indwrk + n*n
	llwrk2 = lwork - indwk2 + 1

	if err = DsytrdSb2st('N', jobz, uplo, n, kd, ab, w, work.Off(inde-1), work.Off(indhous-1), lhtrd, work.Off(indwrk-1), llwork); err != nil {
		panic(err)
	}

	//     For eigenvalues only, call DSTERF.  For eigenvectors, call SSTEDC.
	if !wantz {
		if info, err = Dsterf(n, w, work.Off(inde-1)); err != nil {
			panic(err)
		}
	} else {
		if info, err = Dstedc('I', n, w, work.Off(inde-1), work.MatrixOff(indwrk-1, n, opts), work.Off(indwk2-1), llwrk2, iwork, liwork); err != nil {
			panic(err)
		}
		if err = goblas.Dgemm(NoTrans, NoTrans, n, n, n, one, z, work.MatrixOff(indwrk-1, n, opts), zero, work.MatrixOff(indwk2-1, n, opts)); err != nil {
			panic(err)
		}
		Dlacpy(Full, n, n, work.MatrixOff(indwk2-1, n, opts), z)
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
	if iscale == 1 {
		goblas.Dscal(n, one/sigma, w.Off(0, 1))
	}

	work.Set(0, float64(lwmin))
	(*iwork)[0] = liwmin

	return
}
