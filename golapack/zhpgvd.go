package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhpgvd computes all the eigenvalues and, optionally, the eigenvectors
// of a complex generalized Hermitian-definite eigenproblem, of the form
// A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A and
// B are assumed to be Hermitian, stored in packed format, and B is also
// positive definite.
// If eigenvectors are desired, it uses a divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Zhpgvd(itype *int, jobz, uplo byte, n *int, ap, bp *mat.CVector, w *mat.Vector, z *mat.CMatrix, ldz *int, work *mat.CVector, lwork *int, rwork *mat.Vector, lrwork *int, iwork *[]int, liwork, info *int) {
	var lquery, upper, wantz bool
	var trans byte
	var j, liwmin, lrwmin, lwmin, neig int
	var err error
	_ = err

	//     Test the input parameters.
	wantz = jobz == 'V'
	upper = uplo == 'U'
	lquery = ((*lwork) == -1 || (*lrwork) == -1 || (*liwork) == -1)

	(*info) = 0
	if (*itype) < 1 || (*itype) > 3 {
		(*info) = -1
	} else if !(wantz || jobz == 'N') {
		(*info) = -2
	} else if !(upper || uplo == 'L') {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*ldz) < 1 || (wantz && (*ldz) < (*n)) {
		(*info) = -9
	}

	if (*info) == 0 {
		if (*n) <= 1 {
			lwmin = 1
			liwmin = 1
			lrwmin = 1
		} else {
			if wantz {
				lwmin = 2 * (*n)
				lrwmin = 1 + 5*(*n) + 2*powint(*n, 2)
				liwmin = 3 + 5*(*n)
			} else {
				lwmin = (*n)
				lrwmin = (*n)
				liwmin = 1
			}
		}

		work.SetRe(0, float64(lwmin))
		rwork.Set(0, float64(lrwmin))
		(*iwork)[0] = liwmin
		if (*lwork) < lwmin && !lquery {
			(*info) = -11
		} else if (*lrwork) < lrwmin && !lquery {
			(*info) = -13
		} else if (*liwork) < liwmin && !lquery {
			(*info) = -15
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHPGVD"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Form a Cholesky factorization of B.
	Zpptrf(uplo, n, bp, info)
	if (*info) != 0 {
		(*info) = (*n) + (*info)
		return
	}

	//     Transform problem to standard eigenvalue problem and solve.
	Zhpgst(itype, uplo, n, ap, bp, info)
	Zhpevd(jobz, uplo, n, ap, w, z, ldz, work, lwork, rwork, lrwork, iwork, liwork, info)
	lwmin = int(maxf64(float64(lwmin), work.GetRe(0)))
	lrwmin = int(maxf64(float64(lrwmin), rwork.Get(0)))
	liwmin = int(maxf64(float64(liwmin), float64((*iwork)[0])))

	if wantz {
		//        Backtransform eigenvectors to the original problem.
		neig = (*n)
		if (*info) > 0 {
			neig = (*info) - 1
		}
		if (*itype) == 1 || (*itype) == 2 {
			//           For A*x=(lambda)*B*x and A*B*x=(lambda)*x;
			//           backtransform eigenvectors: x = inv(L)**H *y or inv(U)*y
			if upper {
				trans = 'N'
			} else {
				trans = 'C'
			}

			for j = 1; j <= neig; j++ {
				err = goblas.Ztpsv(mat.UploByte(uplo), mat.TransByte(trans), NonUnit, *n, bp, z.CVector(0, j-1), 1)
			}

		} else if (*itype) == 3 {
			//           For B*A*x=(lambda)*x;
			//           backtransform eigenvectors: x = L*y or U**H *y
			if upper {
				trans = 'C'
			} else {
				trans = 'N'
			}

			for j = 1; j <= neig; j++ {
				err = goblas.Ztpmv(mat.UploByte(uplo), mat.TransByte(trans), NonUnit, *n, bp, z.CVector(0, j-1), 1)
			}
		}
	}

	work.SetRe(0, float64(lwmin))
	rwork.Set(0, float64(lrwmin))
	(*iwork)[0] = liwmin
}
