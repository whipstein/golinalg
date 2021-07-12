package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsygv2stage computes all the eigenvalues, and optionally, the eigenvectors
// of a real generalized symmetric-definite eigenproblem, of the form
// A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.
// Here A and B are assumed to be symmetric and B is also
// positive definite.
// This routine use the 2stage technique for the reduction to tridiagonal
// which showed higher performance on recent architecture and for large
// sizes N>2000.
func Dsygv2stage(itype *int, jobz, uplo byte, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, w, work *mat.Vector, lwork, info *int) {
	var lquery, upper, wantz bool
	var trans byte
	var one float64
	var ib, kd, lhtrd, lwmin, lwtrd, neig int
	var err error
	_ = err

	one = 1.0

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
	} else if (*lda) < max(1, *n) {
		(*info) = -6
	} else if (*ldb) < max(1, *n) {
		(*info) = -8
	}

	if (*info) == 0 {
		kd = Ilaenv2stage(func() *int { y := 1; return &y }(), []byte("DSYTRD_2STAGE"), []byte{jobz}, n, toPtr(-1), toPtr(-1), toPtr(-1))
		ib = Ilaenv2stage(func() *int { y := 2; return &y }(), []byte("DSYTRD_2STAGE"), []byte{jobz}, n, &kd, toPtr(-1), toPtr(-1))
		lhtrd = Ilaenv2stage(func() *int { y := 3; return &y }(), []byte("DSYTRD_2STAGE"), []byte{jobz}, n, &kd, &ib, toPtr(-1))
		lwtrd = Ilaenv2stage(func() *int { y := 4; return &y }(), []byte("DSYTRD_2STAGE"), []byte{jobz}, n, &kd, &ib, toPtr(-1))
		lwmin = 2*(*n) + lhtrd + lwtrd
		work.Set(0, float64(lwmin))

		if (*lwork) < lwmin && !lquery {
			(*info) = -11
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYGV_2STAGE "), -(*info))
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
	Dsyev2stage(jobz, uplo, n, a, lda, w, work, lwork, info)

	if wantz {
		//        Backtransform eigenvectors to the original problem.
		neig = (*n)
		if (*info) > 0 {
			neig = (*info) - 1
		}
		if (*itype) == 1 || (*itype) == 2 {
			//           For A*x=(lambda)*B*x and A*B*x=(lambda)*x;
			//           backtransform eigenvectors: x = inv(L)**T*y or inv(U)*y
			if upper {
				trans = 'N'
			} else {
				trans = 'T'
			}

			err = goblas.Dtrsm(Left, mat.UploByte(uplo), mat.TransByte(trans), NonUnit, *n, neig, one, b, a)

		} else if (*itype) == 3 {
			//           For B*A*x=(lambda)*x;
			//           backtransform eigenvectors: x = L*y or U**T*y
			if upper {
				trans = 'T'
			} else {
				trans = 'N'
			}

			err = goblas.Dtrmm(Left, mat.UploByte(uplo), mat.TransByte(trans), NonUnit, *n, neig, one, b, a)
		}
	}

	work.Set(0, float64(lwmin))
}
