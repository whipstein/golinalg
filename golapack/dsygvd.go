package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsygvd computes all the eigenvalues, and optionally, the eigenvectors
// of a real generalized symmetric-definite eigenproblem, of the form
// A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A and
// B are assumed to be symmetric and B is also positive definite.
// If eigenvectors are desired, it uses a divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Dsygvd(itype *int, jobz, uplo byte, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, w, work *mat.Vector, lwork *int, iwork *[]int, liwork, info *int) {
	var lquery, upper, wantz bool
	var trans byte
	var one float64
	var liopt, liwmin, lopt, lwmin int
	var err error
	_ = err

	one = 1.0

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
		lwmin = 1 + 6*(*n) + 2*int(math.Pow(float64(*n), 2))
	} else {
		liwmin = 1
		lwmin = 2*(*n) + 1
	}
	lopt = lwmin
	liopt = liwmin
	if (*itype) < 1 || (*itype) > 3 {
		(*info) = -1
	} else if !(wantz || jobz == 'N') {
		(*info) = -2
	} else if !(upper || uplo == 'L') {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*lda) < max(1, *n) {
		(*info) = -6
	} else if (*ldb) < max(1, *n) {
		(*info) = -8
	}

	if (*info) == 0 {
		work.Set(0, float64(lopt))
		(*iwork)[0] = liopt

		if (*lwork) < lwmin && !lquery {
			(*info) = -11
		} else if (*liwork) < liwmin && !lquery {
			(*info) = -13
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYGVD"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Form a Cholesky factorization of B.
	Dpotrf(uplo, n, b, ldb, info)
	if (*info) != 0 {
		(*info) = (*n) + (*info)
		return
	}

	//     Transform problem to standard eigenvalue problem and solve.
	Dsygst(itype, uplo, n, a, lda, b, ldb, info)
	Dsyevd(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info)
	lopt = max(lopt, int(work.Get(0)))
	liopt = max(liopt, (*iwork)[0])

	if wantz && (*info) == 0 {
		//        Backtransform eigenvectors to the original problem.
		if (*itype) == 1 || (*itype) == 2 {
			//           For A*x=(lambda)*B*x and A*B*x=(lambda)*x;
			//           backtransform eigenvectors: x = inv(L)**T*y or inv(U)*y
			if upper {
				trans = 'N'
			} else {
				trans = 'T'
			}

			err = goblas.Dtrsm(Left, mat.UploByte(uplo), mat.TransByte(trans), NonUnit, *n, *n, one, b, a)

		} else if (*itype) == 3 {
			//           For B*A*x=(lambda)*x;
			//           backtransform eigenvectors: x = L*y or U**T*y
			if upper {
				trans = 'T'
			} else {
				trans = 'N'
			}

			err = goblas.Dtrmm(Left, mat.UploByte(uplo), mat.TransByte(trans), NonUnit, *n, *n, one, b, a)
		}
	}

	work.Set(0, float64(lopt))
	(*iwork)[0] = liopt
}
