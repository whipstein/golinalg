package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsbgvd computes all the eigenvalues, and optionally, the eigenvectors
// of a real generalized symmetric-definite banded eigenproblem, of the
// form A*x=(lambda)*B*x.  Here A and B are assumed to be symmetric and
// banded, and B is also positive definite.  If eigenvectors are
// desired, it uses a divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Dsbgvd(jobz byte, uplo mat.MatUplo, n, ka, kb int, ab, bb *mat.Matrix, w *mat.Vector, z *mat.Matrix, work *mat.Vector, lwork int, iwork *[]int, liwork int) (info int, err error) {
	var lquery, upper, wantz bool
	var vect byte
	var one, zero float64
	var inde, indwk2, indwrk, liwmin, llwrk2, lwmin int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	wantz = jobz == 'V'
	upper = uplo == Upper
	lquery = (lwork == -1 || liwork == -1)

	if n <= 1 {
		liwmin = 1
		lwmin = 1
	} else if wantz {
		liwmin = 3 + 5*n
		lwmin = 1 + 5*n + 2*pow(n, 2)
	} else {
		liwmin = 1
		lwmin = 2 * n
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
		err = fmt.Errorf("kb < 0 || kb > ka: kb=%v, ka=%v", kb, ka)
	} else if ab.Rows < ka+1 {
		err = fmt.Errorf("ab.Rows < ka+1: ab.Rows=%v, ka=%v", ab.Rows, ka)
	} else if bb.Rows < kb+1 {
		err = fmt.Errorf("bb.Rows < kb+1: bb.Rows=%v, kb=%v", bb.Rows, kb)
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
		gltest.Xerbla2("Dsbgvd", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Form a split Cholesky factorization of B.
	if info, err = Dpbstf(uplo, n, kb, bb); err != nil {
		panic(err)
	}
	if info != 0 {
		info = n + info
		return
	}

	//     Transform problem to standard eigenvalue problem.
	inde = 1
	indwrk = inde + n
	indwk2 = indwrk + n*n
	llwrk2 = lwork - indwk2 + 1
	if err = Dsbgst(jobz, uplo, n, ka, kb, ab, bb, z, work); err != nil {
		panic(err)
	}

	//     Reduce to tridiagonal form.
	if wantz {
		vect = 'U'
	} else {
		vect = 'N'
	}
	if err = Dsbtrd(vect, uplo, n, ka, ab, w, work.Off(inde-1), z, work.Off(indwrk-1)); err != nil {
		panic(err)
	}

	//     For eigenvalues only, call DSTERF. For eigenvectors, call SSTEDC.
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

	work.Set(0, float64(lwmin))
	(*iwork)[0] = liwmin

	return
}
