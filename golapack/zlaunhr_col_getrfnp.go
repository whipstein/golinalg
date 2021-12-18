package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// ZlaunhrColGetrfnp computes the modified LU factorization without
// pivoting of a complex general M-by-N matrix A. The factorization has
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
// reconstruction routine ZUNHR_COL. In ZUNHR_COL, this routine is
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
// this routine calls the recursive routine ZLAUNHR_COL_GETRFNP2.
//
// [1] "Reconstructing Householder vectors from tall-skinny QR",
//     G. Ballard, J. Demmel, L. Grigori, M. Jacquelin, H.D. Nguyen,
//     E. Solomonik, J. Parallel Distrib. Comput.,
//     vol. 85, pp. 3-31, 2015.
func ZlaunhrColGetrfnp(m, n int, a *mat.CMatrix, d *mat.CVector) (err error) {
	var cone complex128
	var j, jb, nb int

	cone = (1.0 + 0.0*1i)

	//     Test the input parameters.
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("ZlaunhrColGetrfnp", err)
		return
	}

	//     Quick return if possible
	if min(m, n) == 0 {
		return
	}

	//     Determine the block size for this environment.
	nb = Ilaenv(1, "ZlaunhrColGetrfnp", []byte{' '}, m, n, -1, -1)
	if nb <= 1 || nb >= min(m, n) {
		//        Use unblocked code.
		if err = ZlaunhrColGetrfnp2(m, n, a, d); err != nil {
			panic(err)
		}
	} else {
		//        Use blocked code.
		for j = 1; j <= min(m, n); j += nb {
			jb = min(min(m, n)-j+1, nb)

			//           Factor diagonal and subdiagonal blocks.
			if err = ZlaunhrColGetrfnp2(m-j+1, jb, a.Off(j-1, j-1), d.Off(j-1)); err != nil {
				panic(err)
			}

			if j+jb <= n {
				//              Compute block row of U.
				if err = a.Off(j-1, j+jb-1).Trsm(Left, Lower, NoTrans, Unit, jb, n-j-jb+1, cone, a.Off(j-1, j-1)); err != nil {
					panic(err)
				}
				if j+jb <= m {
					//                 Update trailing submatrix.
					if err = a.Off(j+jb-1, j+jb-1).Gemm(NoTrans, NoTrans, m-j-jb+1, n-j-jb+1, jb, -cone, a.Off(j+jb-1, j-1), a.Off(j-1, j+jb-1), cone); err != nil {
						panic(err)
					}
				}
			}
		}
	}

	return
}
