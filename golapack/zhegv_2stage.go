package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhegv2stage computes all the eigenvalues, and optionally, the eigenvectors
// of a complex generalized Hermitian-definite eigenproblem, of the form
// A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.
// Here A and B are assumed to be Hermitian and B is also
// positive definite.
// This routine use the 2stage technique for the reduction to tridiagonal
// which showed higher performance on recent architecture and for large
// sizes N>2000.
func Zhegv2stage(itype *int, jobz, uplo byte, n *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, w *mat.Vector, work *mat.CVector, lwork *int, rwork *mat.Vector, info *int) {
	var lquery, upper, wantz bool
	var trans byte
	var one complex128
	var ib, kd, lhtrd, lwmin, lwtrd, neig int
	var err error
	_ = err

	one = (1.0 + 0.0*1i)

	//     Test the input parameters.
	wantz = jobz == 'V'
	upper = uplo == 'U'
	lquery = ((*lwork) == -1)

	(*info) = 0
	if (*itype) < 1 || (*itype) > 3 {
		(*info) = -1
	} else if jobz != 'N' {
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
		kd = Ilaenv2stage(func() *int { y := 1; return &y }(), []byte("ZHETRD_2STAGE"), []byte{jobz}, n, toPtr(-1), toPtr(-1), toPtr(-1))
		ib = Ilaenv2stage(func() *int { y := 2; return &y }(), []byte("ZHETRD_2STAGE"), []byte{jobz}, n, &kd, toPtr(-1), toPtr(-1))
		lhtrd = Ilaenv2stage(func() *int { y := 3; return &y }(), []byte("ZHETRD_2STAGE"), []byte{jobz}, n, &kd, &ib, toPtr(-1))
		lwtrd = Ilaenv2stage(func() *int { y := 4; return &y }(), []byte("ZHETRD_2STAGE"), []byte{jobz}, n, &kd, &ib, toPtr(-1))
		lwmin = (*n) + lhtrd + lwtrd
		work.SetRe(0, float64(lwmin))

		if (*lwork) < lwmin && !lquery {
			(*info) = -11
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHEGV_2STAGE "), -(*info))
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
	Zheev2stage(jobz, uplo, n, a, lda, w, work, lwork, rwork, info)

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

			err = goblas.Ztrsm(Left, mat.UploByte(uplo), mat.TransByte(trans), NonUnit, *n, neig, one, b, *ldb, a, *lda)

		} else if (*itype) == 3 {
			//           For B*A*x=(lambda)*x;
			//           backtransform eigenvectors: x = L*y or U**H *y
			if upper {
				trans = 'C'
			} else {
				trans = 'N'
			}

			err = goblas.Ztrmm(Left, mat.UploByte(uplo), mat.TransByte(trans), NonUnit, *n, neig, one, b, *ldb, a, *lda)
		}
	}

	work.SetRe(0, float64(lwmin))
}
