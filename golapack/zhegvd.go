package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zhegvd computes all the eigenvalues, and optionally, the eigenvectors
// of a complex generalized Hermitian-definite eigenproblem, of the form
// A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A and
// B are assumed to be Hermitian and B is also positive definite.
// If eigenvectors are desired, it uses a divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Zhegvd(itype *int, jobz, uplo byte, n *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, w *mat.Vector, work *mat.CVector, lwork *int, rwork *mat.Vector, lrwork *int, iwork *[]int, liwork, info *int) {
	var lquery, upper, wantz bool
	var trans byte
	var cone complex128
	var liopt, liwmin, lopt, lropt, lrwmin, lwmin int

	cone = (1.0 + 0.0*1i)

	//     Test the input parameters.
	wantz = jobz == 'V'
	upper = uplo == 'U'
	lquery = ((*lwork) == -1 || (*lrwork) == -1 || (*liwork) == -1)

	(*info) = 0
	if (*n) <= 1 {
		lwmin = 1
		lrwmin = 1
		liwmin = 1
	} else if wantz {
		lwmin = 2*(*n) + (*n)*(*n)
		lrwmin = 1 + 5*(*n) + 2*(*n)*(*n)
		liwmin = 3 + 5*(*n)
	} else {
		lwmin = (*n) + 1
		lrwmin = (*n)
		liwmin = 1
	}
	lopt = lwmin
	lropt = lrwmin
	liopt = liwmin
	if (*itype) < 1 || (*itype) > 3 {
		(*info) = -1
	} else if !(wantz || jobz == 'N') {
		(*info) = -2
	} else if !(upper || uplo == 'L') {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*lda) < maxint(1, *n) {
		(*info) = -6
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -8
	}

	if (*info) == 0 {
		work.SetRe(0, float64(lopt))
		rwork.Set(0, float64(lropt))
		(*iwork)[0] = liopt
		//
		if (*lwork) < lwmin && !lquery {
			(*info) = -11
		} else if (*lrwork) < lrwmin && !lquery {
			(*info) = -13
		} else if (*liwork) < liwmin && !lquery {
			(*info) = -15
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHEGVD"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Form a Cholesky factorization of B.
	Zpotrf(uplo, n, b, ldb, info)
	if (*info) != 0 {
		(*info) = (*n) + (*info)
		return
	}

	//     Transform problem to standard eigenvalue problem and solve.
	Zhegst(itype, uplo, n, a, lda, b, ldb, info)
	Zheevd(jobz, uplo, n, a, lda, w, work, lwork, rwork, lrwork, iwork, liwork, info)
	lopt = int(maxf64(float64(lopt), work.GetRe(0)))
	lropt = int(maxf64(float64(lropt), rwork.Get(0)))
	liopt = int(maxf64(float64(liopt), float64((*iwork)[0])))

	if wantz && (*info) == 0 {
		//        Backtransform eigenvectors to the original problem.
		if (*itype) == 1 || (*itype) == 2 {
			//           For A*x=(lambda)*B*x and A*B*x=(lambda)*x;
			//           backtransform eigenvectors: x = inv(L)**H *y or inv(U)*y
			if upper {
				trans = 'N'
			} else {
				trans = 'C'
			}

			goblas.Ztrsm(Left, mat.UploByte(uplo), mat.TransByte(trans), NonUnit, n, n, &cone, b, ldb, a, lda)

		} else if (*itype) == 3 {
			//           For B*A*x=(lambda)*x;
			//           backtransform eigenvectors: x = L*y or U**H *y
			if upper {
				trans = 'C'
			} else {
				trans = 'N'
			}

			goblas.Ztrmm(Left, mat.UploByte(uplo), mat.TransByte(trans), NonUnit, n, n, &cone, b, ldb, a, lda)
		}
	}

	work.SetRe(0, float64(lopt))
	rwork.Set(0, float64(lropt))
	(*iwork)[0] = liopt
}
