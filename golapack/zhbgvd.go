package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhbgvd computes all the eigenvalues, and optionally, the eigenvectors
// of a complex generalized Hermitian-definite banded eigenproblem, of
// the form A*x=(lambda)*B*x. Here A and B are assumed to be Hermitian
// and banded, and B is also positive definite.  If eigenvectors are
// desired, it uses a divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Zhbgvd(jobz byte, uplo mat.MatUplo, n, ka, kb int, ab, bb *mat.CMatrix, w *mat.Vector, z *mat.CMatrix, work *mat.CVector, lwork int, rwork *mat.Vector, lrwork int, iwork *[]int, liwork int) (info int, err error) {
	var lquery, upper, wantz bool
	var vect byte
	var cone, czero complex128
	var inde, indwk2, indwrk, liwmin, llrwk, llwk2, lrwmin, lwmin int

	cone = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	wantz = jobz == 'V'
	upper = uplo == Upper
	lquery = (lwork == -1 || lrwork == -1 || liwork == -1)

	if n <= 1 {
		lwmin = 1 + n
		lrwmin = 1 + n
		liwmin = 1
	} else if wantz {
		lwmin = 2 * pow(n, 2)
		lrwmin = 1 + 5*n + 2*pow(n, 2)
		liwmin = 3 + 5*n
	} else {
		lwmin = n
		lrwmin = n
		liwmin = 1
	}
	if !(wantz || jobz == 'N') {
		err = fmt.Errorf("!(wantz || jobz == 'N'): jobz='%c'", jobz)
	} else if !(upper || uplo == Lower) {
		err = fmt.Errorf("!(upper || uplo == Lower): uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if ka < 0 {
		err = fmt.Errorf("ka < 0: ka=%v", ka)
	} else if kb < 0 || kb > ka {
		err = fmt.Errorf("kb < 0 || kb > ka: ka=%v, kb=%v", ka, kb)
	} else if ab.Rows < ka+1 {
		err = fmt.Errorf("ab.Rows < ka+1: ab.Rows=%v, ka=%v", ab.Rows, ka)
	} else if bb.Rows < kb+1 {
		err = fmt.Errorf("bb.Rows < kb+1: bb.Rows=%v, kb=%v", bb.Rows, kb)
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
		gltest.Xerbla2("Zhbgvd", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Form a split Cholesky factorization of B.
	if info, err = Zpbstf(uplo, n, kb, bb); err != nil {
		panic(err)
	}
	if info != 0 {
		info = n + info
		return
	}

	//     Transform problem to standard eigenvalue problem.
	inde = 1
	indwrk = inde + n
	indwk2 = 1 + n*n
	llwk2 = lwork - indwk2 + 2
	llrwk = lrwork - indwrk + 2
	if err = Zhbgst(jobz, uplo, n, ka, kb, ab, bb, z, work, rwork); err != nil {
		panic(err)
	}

	//     Reduce Hermitian band matrix to tridiagonal form.
	if wantz {
		vect = 'U'
	} else {
		vect = 'N'
	}
	if err = Zhbtrd(vect, uplo, n, ka, ab, w, rwork.Off(inde-1), z, work); err != nil {
		panic(err)
	}

	//     For eigenvalues only, call DSTERF.  For eigenvectors, call ZSTEDC.
	if !wantz {
		if info, err = Dsterf(n, w, rwork.Off(inde-1)); err != nil {
			panic(err)
		}
	} else {
		if info, err = Zstedc('I', n, w, rwork.Off(inde-1), work.CMatrix(n, opts), work.Off(indwk2-1), llwk2, rwork.Off(indwrk-1), llrwk, iwork, liwork); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, NoTrans, n, n, n, cone, z, work.CMatrix(n, opts), czero, work.CMatrixOff(indwk2-1, n, opts)); err != nil {
			panic(err)
		}
		Zlacpy(Full, n, n, work.CMatrixOff(indwk2-1, n, opts), z)
	}

	work.SetRe(0, float64(lwmin))
	rwork.Set(0, float64(lrwmin))
	(*iwork)[0] = liwmin

	return
}
