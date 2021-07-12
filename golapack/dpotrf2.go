package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpotrf2 computes the Cholesky factorization of a real symmetric
// positive definite matrix A using the recursive algorithm.
//
// The factorization has the form
//    A = U**T * U,  if UPLO = 'U', or
//    A = L  * L**T,  if UPLO = 'L',
// where U is an upper triangular matrix and L is lower triangular.
//
// This is the recursive version of the algorithm. It divides
// the matrix into four submatrices:
//
//        [  A11 | A12  ]  where A11 is n1 by n1 and A22 is n2 by n2
//    A = [ -----|----- ]  with n1 = n/2
//        [  A21 | A22  ]       n2 = n-n1
//
// The subroutine calls itself to factor A11. Update and scale A21
// or A12, update A22 then calls itself to factor A22.
func Dpotrf2(uplo byte, n *int, a *mat.Matrix, lda, info *int) {
	var upper bool
	var one, zero float64
	var iinfo, n1, n2 int
	var err error
	_ = err

	one = 1.0
	zero = 0.0

	//     Test the input parameters
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *n) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DPOTRF2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     N=1 case
	if (*n) == 1 {
		//        Test for non-positive-definiteness
		if a.Get(0, 0) <= zero || Disnan(int(a.Get(0, 0))) {
			(*info) = 1
			return
		}

		//        Factor
		a.Set(0, 0, math.Sqrt(a.Get(0, 0)))

		//     Use recursive code
	} else {
		n1 = (*n) / 2
		n2 = (*n) - n1

		//        Factor A11
		Dpotrf2(uplo, &n1, a, lda, &iinfo)
		if iinfo != 0 {
			(*info) = iinfo
			return
		}

		//        Compute the Cholesky factorization A = U**T*U
		if upper {
			//           Update and scale A12
			err = goblas.Dtrsm(mat.Left, mat.Upper, mat.Trans, mat.NonUnit, n1, n2, one, a, a.Off(0, n1))

			//           Update and factor A22
			err = goblas.Dsyrk(mat.UploByte(uplo), mat.Trans, n2, n1, -one, a.Off(0, n1), one, a.Off(n1, n1))
			Dpotrf2(uplo, &n2, a.Off(n1, n1), lda, &iinfo)
			if iinfo != 0 {
				(*info) = iinfo + n1
				return
			}

			//        Compute the Cholesky factorization A = L*L**T
		} else {
			//           Update and scale A21
			err = goblas.Dtrsm(mat.Right, mat.Lower, mat.Trans, mat.NonUnit, n2, n1, one, a, a.Off(n1, 0))

			//           Update and factor A22
			err = goblas.Dsyrk(mat.UploByte(uplo), mat.NoTrans, n2, n1, -one, a.Off(n1, 0), one, a.Off(n1, n1))
			Dpotrf2(uplo, &n2, a.Off(n1, n1), lda, &iinfo)
			if iinfo != 0 {
				(*info) = iinfo + n1
				return
			}
		}
	}
}
