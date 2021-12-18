package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgetrf computes an LU factorization of a general M-by-N matrix A
// using partial pivoting with row interchanges.
//
// The factorization has the form
//    A = P * L * U
// where P is a permutation matrix, L is lower triangular with unit
// diagonal elements (lower trapezoidal if m > n), and U is upper
// triangular (upper trapezoidal if m < n).
//
// This is the right-looking Level 3 BLAS version of the algorithm.
func Dgetrf(m, n int, a *mat.Matrix, ipiv *[]int) (info int, err error) {
	var one float64
	var i, iinfo, j, jb, nb int

	one = 1.0

	//     Test the input parameters.
	info = 0
	if m < 0 {
		info = -1
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		info = -2
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		info = -4
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("Dgetrf", err)
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 {
		return
	}

	//     Determine the block size for this environment.
	nb = Ilaenv(1, "Dgetrf", []byte{' '}, m, n, -1, -1)
	if nb <= 1 || nb >= min(m, n) {
		//        Use unblocked code.
		info, err = Dgetrf2(m, n, a, ipiv)
	} else {
		//        Use blocked code.
		for j = 1; j <= min(m, n); j += nb {
			jb = min(min(m, n)-j+1, nb)

			//           Factor diagonal and subdiagonal blocks and test for exact
			//           singularity.
			iinfo, err = Dgetrf2(m-j+1, jb, a.Off(j-1, j-1), toSlice(ipiv, j-1))

			//           Adjust INFO and the pivot indices.
			if info == 0 && iinfo > 0 {
				info = iinfo + j - 1
			}
			for i = j; i <= min(m, j+jb-1); i++ {
				(*ipiv)[i-1] = j - 1 + (*ipiv)[i-1]
			}

			//           Apply interchanges to columns 1:J-1.
			Dlaswp(j-1, a, j, j+jb-1, *ipiv, 1)

			if j+jb <= n {
				//              Apply interchanges to columns J+JB:N.
				Dlaswp(n-j-jb+1, a.Off(0, j+jb-1), j, j+jb-1, *ipiv, 1)

				//              Compute block row of U.
				err = a.Off(j-1, j+jb-1).Trsm(Left, Lower, NoTrans, Unit, jb, n-j-jb+1, one, a.Off(j-1, j-1))
				if j+jb <= m {
					//                 Update trailing submatrix.
					err = a.Off(j+jb-1, j+jb-1).Gemm(NoTrans, NoTrans, m-j-jb+1, n-j-jb+1, jb, -one, a.Off(j+jb-1, j-1), a.Off(j-1, j+jb-1), one)
				}
			}
		}
	}

	return
}
