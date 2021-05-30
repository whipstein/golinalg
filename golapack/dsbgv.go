package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dsbgv computes all the eigenvalues, and optionally, the eigenvectors
// of a real generalized symmetric-definite banded eigenproblem, of
// the form A*x=(lambda)*B*x. Here A and B are assumed to be symmetric
// and banded, and B is also positive definite.
func Dsbgv(jobz, uplo byte, n, ka, kb *int, ab *mat.Matrix, ldab *int, bb *mat.Matrix, ldbb *int, w *mat.Vector, z *mat.Matrix, ldz *int, work *mat.Vector, info *int) {
	var upper, wantz bool
	var vect byte
	var iinfo, inde, indwrk int

	//     Test the input parameters.
	wantz = jobz == 'V'
	upper = uplo == 'U'

	(*info) = 0
	if !(wantz || jobz == 'N') {
		(*info) = -1
	} else if !(upper || uplo == 'L') {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*ka) < 0 {
		(*info) = -4
	} else if (*kb) < 0 || (*kb) > (*ka) {
		(*info) = -5
	} else if (*ldab) < (*ka)+1 {
		(*info) = -7
	} else if (*ldbb) < (*kb)+1 {
		(*info) = -9
	} else if (*ldz) < 1 || (wantz && (*ldz) < (*n)) {
		(*info) = -12
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DSBGV "), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Form a split Cholesky factorization of B.
	Dpbstf(uplo, n, kb, bb, ldbb, info)
	if (*info) != 0 {
		(*info) = (*n) + (*info)
		return
	}

	//     Transform problem to standard eigenvalue problem.
	inde = 1
	indwrk = inde + (*n)
	Dsbgst(jobz, uplo, n, ka, kb, ab, ldab, bb, ldbb, z, ldz, work.Off(indwrk-1), &iinfo)

	//     Reduce to tridiagonal form.
	if wantz {
		vect = 'U'
	} else {
		vect = 'N'
	}
	Dsbtrd(vect, uplo, n, ka, ab, ldab, w, work.Off(inde-1), z, ldz, work.Off(indwrk-1), &iinfo)

	//     For eigenvalues only, call DSTERF.  For eigenvectors, call SSTEQR.
	if !wantz {
		Dsterf(n, w, work.Off(inde-1), info)
	} else {
		Dsteqr(jobz, n, w, work.Off(inde-1), z, ldz, work.Off(indwrk-1), info)
	}
}
