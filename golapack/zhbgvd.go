package golapack

import (
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
func Zhbgvd(jobz, uplo byte, n, ka, kb *int, ab *mat.CMatrix, ldab *int, bb *mat.CMatrix, ldbb *int, w *mat.Vector, z *mat.CMatrix, ldz *int, work *mat.CVector, lwork *int, rwork *mat.Vector, lrwork *int, iwork *[]int, liwork, info *int) {
	var lquery, upper, wantz bool
	var vect byte
	var cone, czero complex128
	var iinfo, inde, indwk2, indwrk, liwmin, llrwk, llwk2, lrwmin, lwmin int
	var err error
	_ = err

	cone = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	wantz = jobz == 'V'
	upper = uplo == 'U'
	lquery = ((*lwork) == -1 || (*lrwork) == -1 || (*liwork) == -1)

	(*info) = 0
	if (*n) <= 1 {
		lwmin = 1 + (*n)
		lrwmin = 1 + (*n)
		liwmin = 1
	} else if wantz {
		lwmin = 2 * powint(*n, 2)
		lrwmin = 1 + 5*(*n) + 2*powint(*n, 2)
		liwmin = 3 + 5*(*n)
	} else {
		lwmin = (*n)
		lrwmin = (*n)
		liwmin = 1
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
		work.SetRe(0, float64(lwmin))
		rwork.Set(0, float64(lrwmin))
		(*iwork)[0] = liwmin

		if (*lwork) < lwmin && !lquery {
			(*info) = -14
		} else if (*lrwork) < lrwmin && !lquery {
			(*info) = -16
		} else if (*liwork) < liwmin && !lquery {
			(*info) = -18
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHBGVD"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Form a split Cholesky factorization of B.
	Zpbstf(uplo, n, kb, bb, ldbb, info)
	if (*info) != 0 {
		(*info) = (*n) + (*info)
		return
	}

	//     Transform problem to standard eigenvalue problem.
	inde = 1
	indwrk = inde + (*n)
	indwk2 = 1 + (*n)*(*n)
	llwk2 = (*lwork) - indwk2 + 2
	llrwk = (*lrwork) - indwrk + 2
	Zhbgst(jobz, uplo, n, ka, kb, ab, ldab, bb, ldbb, z, ldz, work, rwork, &iinfo)

	//     Reduce Hermitian band matrix to tridiagonal form.
	if wantz {
		vect = 'U'
	} else {
		vect = 'N'
	}
	Zhbtrd(vect, uplo, n, ka, ab, ldab, w, rwork.Off(inde-1), z, ldz, work, &iinfo)

	//     For eigenvalues only, call DSTERF.  For eigenvectors, call ZSTEDC.
	if !wantz {
		Dsterf(n, w, rwork.Off(inde-1), info)
	} else {
		Zstedc('I', n, w, rwork.Off(inde-1), work.CMatrix(*n, opts), n, work.Off(indwk2-1), &llwk2, rwork.Off(indwrk-1), &llrwk, iwork, liwork, info)
		err = goblas.Zgemm(NoTrans, NoTrans, *n, *n, *n, cone, z, *ldz, work.CMatrix(*n, opts), *n, czero, work.CMatrixOff(indwk2-1, *n, opts), *n)
		Zlacpy('A', n, n, work.CMatrixOff(indwk2-1, *n, opts), n, z, ldz)
	}

	work.SetRe(0, float64(lwmin))
	rwork.Set(0, float64(lrwmin))
	(*iwork)[0] = liwmin
}
