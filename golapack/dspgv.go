package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dspgv computes all the eigenvalues and, optionally, the eigenvectors
// of a real generalized symmetric-definite eigenproblem, of the form
// A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.
// Here A and B are assumed to be symmetric, stored in packed format,
// and B is also positive definite.
func Dspgv(itype *int, jobz, uplo byte, n *int, ap, bp, w *mat.Vector, z *mat.Matrix, ldz *int, work *mat.Vector, info *int) {
	var upper, wantz bool
	var trans byte
	var j, neig int
	var err error
	_ = err

	//     Test the input parameters.
	wantz = jobz == 'V'
	upper = uplo == 'U'

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
	if (*info) != 0 {
		gltest.Xerbla([]byte("DSPGV "), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Form a Cholesky factorization of B.
	Dpptrf(uplo, n, bp, info)
	if (*info) != 0 {
		(*info) = (*n) + (*info)
		return
	}

	//     Transform problem to standard eigenvalue problem and solve.
	Dspgst(itype, uplo, n, ap, bp, info)
	Dspev(jobz, uplo, n, ap, w, z, ldz, work, info)

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

			for j = 1; j <= neig; j++ {
				err = goblas.Dtpsv(mat.UploByte(uplo), mat.TransByte(trans), NonUnit, *n, bp, z.Vector(0, j-1, 1))
			}

		} else if (*itype) == 3 {
			//           For B*A*x=(lambda)*x;
			//           backtransform eigenvectors: x = L*y or U**T*y
			if upper {
				trans = 'T'
			} else {
				trans = 'N'
			}

			for j = 1; j <= neig; j++ {
				err = goblas.Dtpmv(mat.UploByte(uplo), mat.TransByte(trans), NonUnit, *n, bp, z.Vector(0, j-1, 1))
			}
		}
	}
}
