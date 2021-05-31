package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlaorhrcolgetrfnp computes the modified LU factorization without
// pivoting of a real general M-by-N matrix A. The factorization has
// the form:
//
//     A - S = L * U,
//
// where:
//    S is a m-by-n diagonal sign matrix with the diagonal D, so that
//    D(i) = S(i,i), 1 <= i <= min(M,N). The diagonal D is constructed
//    as D(i)=-SIGN(A(i,i)), where A(i,i) is the value after performing
//    i-1 steps of Gaussian elimination. This means that the diagonal
//    element at each step of "modified" Gaussian elimination is
//    at least one in absolute value (so that division-by-zero not
//    not possible during the division by the diagonal element);
//
//    L is a M-by-N lower triangular matrix with unit diagonal elements
//    (lower trapezoidal if M > N);
//
//    and U is a M-by-N upper triangular matrix
//    (upper trapezoidal if M < N).
//
// This routine is an auxiliary routine used in the Householder
// reconstruction routine DORHR_COL. In DORHR_COL, this routine is
// applied to an M-by-N matrix A with orthonormal columns, where each
// element is bounded by one in absolute value. With the choice of
// the matrix S above, one can show that the diagonal element at each
// step of Gaussian elimination is the largest (in absolute value) in
// the column on or below the diagonal, so that no pivoting is required
// for numerical stability [1].
//
// For more details on the Householder reconstruction algorithm,
// including the modified LU factorization, see [1].
//
// This is the blocked right-looking version of the algorithm,
// calling Level 3 BLAS to update the submatrix. To factorize a block,
// this routine calls the recursive routine DLAORHR_COL_GETRFNP2.
//
// [1] "Reconstructing Householder vectors from tall-skinny QR",
//     G. Ballard, J. Demmel, L. Grigori, M. Jacquelin, H.D. Nguyen,
//     E. Solomonik, J. Parallel Distrib. Comput.,
//     vol. 85, pp. 3-31, 2015.
func DlaorhrColGetrfnp(m, n *int, a *mat.Matrix, lda *int, d *mat.Vector, info *int) {
	var one float64
	var iinfo, j, jb, nb int

	one = 1.0

	//     Test the input parameters.
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *m) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLAORHR_COL_GETRFNP"), -(*info))
		return
	}

	//     Quick return if possible
	if minint(*m, *n) == 0 {
		return
	}

	//     Determine the block size for this environment.
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DLAORHR_COL_GETRFNP"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
	if nb <= 1 || nb >= minint(*m, *n) {
		//        Use unblocked code.
		DlaorhrColGetrfnp2(m, n, a, lda, d, info)
	} else {
		//        Use blocked code.
		for j = 1; j <= minint(*m, *n); j += nb {
			jb = minint(minint(*m, *n)-j+1, nb)

			//           Factor diagonal and subdiagonal blocks.
			DlaorhrColGetrfnp2(toPtr((*m)-j+1), &jb, a.Off(j-1, j-1), lda, d.Off(j-1), &iinfo)

			if j+jb <= (*n) {
				//              Compute block row of U.
				goblas.Dtrsm(Left, Lower, NoTrans, Unit, &jb, toPtr((*n)-j-jb+1), &one, a.Off(j-1, j-1), lda, a.Off(j-1, j+jb-1), lda)
				if j+jb <= (*m) {
					//                 Update trailing submatrix.
					goblas.Dgemm(NoTrans, NoTrans, toPtr((*m)-j-jb+1), toPtr((*n)-j-jb+1), &jb, toPtrf64(-one), a.Off(j+jb-1, j-1), lda, a.Off(j-1, j+jb-1), lda, &one, a.Off(j+jb-1, j+jb-1), lda)
				}
			}
		}
	}
}
