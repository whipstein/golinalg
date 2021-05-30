package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zhpgvx computes selected eigenvalues and, optionally, eigenvectors
// of a complex generalized Hermitian-definite eigenproblem, of the form
// A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A and
// B are assumed to be Hermitian, stored in packed format, and B is also
// positive definite.  Eigenvalues and eigenvectors can be selected by
// specifying either a _range of values or a _range of indices for the
// desired eigenvalues.
func Zhpgvx(itype *int, jobz, _range, uplo byte, n *int, ap, bp *mat.CVector, vl, vu *float64, il, iu *int, abstol *float64, m *int, w *mat.Vector, z *mat.CMatrix, ldz *int, work *mat.CVector, rwork *mat.Vector, iwork, ifail *[]int, info *int) {
	var alleig, indeig, upper, valeig, wantz bool
	var trans byte
	var j int

	//     Test the input parameters.
	wantz = jobz == 'V'
	upper = uplo == 'U'
	alleig = _range == 'A'
	valeig = _range == 'V'
	indeig = _range == 'I'

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
	} else {
		if valeig {
			if (*n) > 0 && (*vu) <= (*vl) {
				(*info) = -9
			}
		} else if indeig {
			if (*il) < 1 {
				(*info) = -10
			} else if (*iu) < minint(*n, *il) || (*iu) > (*n) {
				(*info) = -11
			}
		}
	}
	if (*info) == 0 {
		if (*ldz) < 1 || (wantz && (*ldz) < (*n)) {
			(*info) = -16
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHPGVX"), -(*info))
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
	Zhpevx(jobz, _range, uplo, n, ap, vl, vu, il, iu, abstol, m, w, z, ldz, work, rwork, iwork, ifail, info)

	if wantz {
		//        Backtransform eigenvectors to the original problem.
		if (*info) > 0 {
			(*m) = (*info) - 1
		}
		if (*itype) == 1 || (*itype) == 2 {
			//           For A*x=(lambda)*B*x and A*B*x=(lambda)*x;
			//           backtransform eigenvectors: x = inv(L)**H *y or inv(U)*y
			if upper {
				trans = 'N'
			} else {
				trans = 'C'
			}

			for j = 1; j <= (*m); j++ {
				goblas.Ztpsv(mat.UploByte(uplo), mat.TransByte(trans), NonUnit, n, bp, z.CVector(0, j-1), func() *int { y := 1; return &y }())
			}

		} else if (*itype) == 3 {
			//           For B*A*x=(lambda)*x;
			//           backtransform eigenvectors: x = L*y or U**H *y
			if upper {
				trans = 'C'
			} else {
				trans = 'N'
			}

			for j = 1; j <= (*m); j++ {
				goblas.Ztpmv(mat.UploByte(uplo), mat.TransByte(trans), NonUnit, n, bp, z.CVector(0, j-1), func() *int { y := 1; return &y }())
			}
		}
	}
}
