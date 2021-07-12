package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsygvx computes selected eigenvalues, and optionally, eigenvectors
// of a real generalized symmetric-definite eigenproblem, of the form
// A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A
// and B are assumed to be symmetric and B is also positive definite.
// Eigenvalues and eigenvectors can be selected by specifying either a
// _range of values or a _range of indices for the desired eigenvalues.
func Dsygvx(itype *int, jobz, _range, uplo byte, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, vl, vu *float64, il, iu *int, abstol *float64, m *int, w *mat.Vector, z *mat.Matrix, ldz *int, work *mat.Vector, lwork *int, iwork, ifail *[]int, info *int) {
	var alleig, indeig, lquery, upper, valeig, wantz bool
	var trans byte
	var one float64
	var lwkmin, lwkopt, nb int
	var err error
	_ = err

	one = 1.0

	//     Test the input parameters.
	upper = uplo == 'U'
	wantz = jobz == 'V'
	alleig = _range == 'A'
	valeig = _range == 'V'
	indeig = _range == 'I'
	lquery = ((*lwork) == -1)

	(*info) = 0
	if (*itype) < 1 || (*itype) > 3 {
		(*info) = -1
	} else if !(wantz || jobz == 'N') {
		(*info) = -2
	} else if !(alleig || valeig || indeig) {
		(*info) = -3
	} else if !(upper || uplo == 'L') {
		(*info) = -4
	} else if (*n) < 0 {
		(*info) = -5
	} else if (*lda) < max(1, *n) {
		(*info) = -7
	} else if (*ldb) < max(1, *n) {
		(*info) = -9
	} else {
		if valeig {
			if (*n) > 0 && (*vu) <= (*vl) {
				(*info) = -11
			}
		} else if indeig {
			if (*il) < 1 || (*il) > max(1, *n) {
				(*info) = -12
			} else if (*iu) < min(*n, *il) || (*iu) > (*n) {
				(*info) = -13
			}
		}
	}
	if (*info) == 0 {
		if (*ldz) < 1 || (wantz && (*ldz) < (*n)) {
			(*info) = -18
		}
	}

	if (*info) == 0 {
		lwkmin = max(1, 8*(*n))
		nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DSYTRD"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))
		lwkopt = max(lwkmin, (nb+3)*(*n))
		work.Set(0, float64(lwkopt))

		if (*lwork) < lwkmin && !lquery {
			(*info) = -20
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYGVX"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	(*m) = 0
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
	Dsyevx(jobz, _range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, work, lwork, iwork, ifail, info)

	if wantz {
		//        Backtransform eigenvectors to the original problem.
		if (*info) > 0 {
			(*m) = (*info) - 1
		}
		if (*itype) == 1 || (*itype) == 2 {
			//           For A*x=(lambda)*B*x and A*B*x=(lambda)*x;
			//           backtransform eigenvectors: x = inv(L)**T*y or inv(U)*y
			if upper {
				trans = 'N'
			} else {
				trans = 'T'
			}

			err = goblas.Dtrsm(Left, mat.UploByte(uplo), mat.TransByte(trans), NonUnit, *n, *m, one, b, z)

		} else if (*itype) == 3 {
			//           For B*A*x=(lambda)*x;
			//           backtransform eigenvectors: x = L*y or U**T*y
			if upper {
				trans = 'T'
			} else {
				trans = 'N'
			}

			err = goblas.Dtrmm(Left, mat.UploByte(uplo), mat.TransByte(trans), NonUnit, *n, *m, one, b, z)
		}
	}

	//     Set WORK(1) to optimal workspace size.
	work.Set(0, float64(lwkopt))
}
