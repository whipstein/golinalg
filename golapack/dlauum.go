package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlauum computes the product U * U**T or L**T * L, where the triangular
// factor U or L is stored in the upper or lower triangular part of
// the array A.
//
// If UPLO = 'U' or 'u' then the upper triangle of the result is stored,
// overwriting the factor U in A.
// If UPLO = 'L' or 'l' then the lower triangle of the result is stored,
// overwriting the factor L in A.
//
// This is the blocked form of the algorithm, calling Level 3 BLAS.
func Dlauum(uplo mat.MatUplo, n int, a *mat.Matrix) (err error) {
	var upper bool
	var one float64
	var i, ib, nb int

	one = 1.0

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dlauum", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Determine the block size for this environment.
	nb = Ilaenv(1, "Dlauum", []byte{uplo.Byte()}, n, -1, -1, -1)

	if nb <= 1 || nb >= n {
		//        Use unblocked code
		if err = Dlauu2(uplo, n, a); err != nil {
			panic(err)
		}
	} else {
		//        Use blocked code
		if upper {
			//           Compute the product U * U**T.
			for i = 1; i <= n; i += nb {
				ib = min(nb, n-i+1)
				if err = a.Off(0, i-1).Trmm(mat.Right, mat.Upper, mat.Trans, mat.NonUnit, i-1, ib, one, a.Off(i-1, i-1)); err != nil {
					panic(err)
				}
				if err = Dlauu2(Upper, ib, a.Off(i-1, i-1)); err != nil {
					panic(err)
				}
				if i+ib <= n {
					if err = a.Off(0, i-1).Gemm(mat.NoTrans, mat.Trans, i-1, ib, n-i-ib+1, one, a.Off(0, i+ib-1), a.Off(i-1, i+ib-1), one); err != nil {
						panic(err)
					}
					if err = a.Off(i-1, i-1).Syrk(mat.Upper, mat.NoTrans, ib, n-i-ib+1, one, a.Off(i-1, i+ib-1), one); err != nil {
						panic(err)
					}
				}
			}
		} else {
			//           Compute the product L**T * L.
			for i = 1; i <= n; i += nb {
				ib = min(nb, n-i+1)
				if err = a.Off(i-1, 0).Trmm(mat.Left, mat.Lower, mat.Trans, mat.NonUnit, ib, i-1, one, a.Off(i-1, i-1)); err != nil {
					panic(err)
				}
				if err = Dlauu2(Lower, ib, a.Off(i-1, i-1)); err != nil {
					panic(err)
				}
				if i+ib <= n {
					if err = a.Off(i-1, 0).Gemm(mat.Trans, mat.NoTrans, ib, i-1, n-i-ib+1, one, a.Off(i+ib-1, i-1), a.Off(i+ib-1, 0), one); err != nil {
						panic(err)
					}
					if err = a.Off(i-1, i-1).Syrk(mat.Lower, mat.Trans, ib, n-i-ib+1, one, a.Off(i+ib-1, i-1), one); err != nil {
						panic(err)
					}
				}
			}
		}
	}

	return
}
