package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhbgv computes all the eigenvalues, and optionally, the eigenvectors
// of a complex generalized Hermitian-definite banded eigenproblem, of
// the form A*x=(lambda)*B*x. Here A and B are assumed to be Hermitian
// and banded, and B is also positive definite.
func Zhbgv(jobz byte, uplo mat.MatUplo, n, ka, kb int, ab, bb *mat.CMatrix, w *mat.Vector, z *mat.CMatrix, work *mat.CVector, rwork *mat.Vector) (info int, err error) {
	var upper, wantz bool
	var vect byte
	var inde, indwrk int

	//     Test the input parameters.
	wantz = jobz == 'V'
	upper = uplo == Upper

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
	if err != nil {
		gltest.Xerbla2("Zhbgv", err)
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
	if err = Zhbgst(jobz, uplo, n, ka, kb, ab, bb, z, work, rwork.Off(indwrk-1)); err != nil {
		panic(err)
	}

	//     Reduce to tridiagonal form.
	if wantz {
		vect = 'U'
	} else {
		vect = 'N'
	}
	if err = Zhbtrd(vect, uplo, n, ka, ab, w, rwork.Off(inde-1), z, work); err != nil {
		panic(err)
	}

	//     For eigenvalues only, call DSTERF.  For eigenvectors, call ZSTEQR.
	if !wantz {
		if info, err = Dsterf(n, w, rwork.Off(inde-1)); err != nil {
			panic(err)
		}
	} else {
		if info, err = Zsteqr(jobz, n, w, rwork.Off(inde-1), z, rwork.Off(indwrk-1)); err != nil {
			panic(err)
		}
	}

	return
}
