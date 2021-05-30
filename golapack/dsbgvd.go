package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
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
func Dsbgvd(jobz, uplo byte, n, ka, kb *int, ab *mat.Matrix, ldab *int, bb *mat.Matrix, ldbb *int, w *mat.Vector, z *mat.Matrix, ldz *int, work *mat.Vector, lwork *int, iwork *[]int, liwork, info *int) {
	var lquery, upper, wantz bool
	var vect byte
	var one, zero float64
	var iinfo, inde, indwk2, indwrk, liwmin, llwrk2, lwmin int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	wantz = jobz == 'V'
	upper = uplo == 'U'
	lquery = ((*lwork) == -1 || (*liwork) == -1)

	(*info) = 0
	if (*n) <= 1 {
		liwmin = 1
		lwmin = 1
	} else if wantz {
		liwmin = 3 + 5*(*n)
		lwmin = 1 + 5*(*n) + 2*int(math.Pow(float64(*n), 2))
	} else {
		liwmin = 1
		lwmin = 2 * (*n)
	}

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

	if (*info) == 0 {
		work.Set(0, float64(lwmin))
		(*iwork)[0] = liwmin

		if (*lwork) < lwmin && !lquery {
			(*info) = -14
		} else if (*liwork) < liwmin && !lquery {
			(*info) = -16
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSBGVD"), -(*info))
		return
	} else if lquery {
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
	indwk2 = indwrk + (*n)*(*n)
	llwrk2 = (*lwork) - indwk2 + 1
	Dsbgst(jobz, uplo, n, ka, kb, ab, ldab, bb, ldbb, z, ldz, work, &iinfo)

	//     Reduce to tridiagonal form.
	if wantz {
		vect = 'U'
	} else {
		vect = 'N'
	}
	Dsbtrd(vect, uplo, n, ka, ab, ldab, w, work.Off(inde-1), z, ldz, work.Off(indwrk-1), &iinfo)

	//     For eigenvalues only, call DSTERF. For eigenvectors, call SSTEDC.
	if !wantz {
		Dsterf(n, w, work.Off(inde-1), info)
	} else {
		Dstedc('I', n, w, work.Off(inde-1), work.MatrixOff(indwrk-1, *n, opts), n, work.Off(indwk2-1), &llwrk2, iwork, liwork, info)
		goblas.Dgemm(NoTrans, NoTrans, n, n, n, &one, z, ldz, work.MatrixOff(indwrk-1, *n, opts), n, &zero, work.MatrixOff(indwk2-1, *n, opts), n)
		Dlacpy('A', n, n, work.MatrixOff(indwk2-1, *n, opts), n, z, ldz)
	}

	work.Set(0, float64(lwmin))
	(*iwork)[0] = liwmin
}
