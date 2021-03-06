package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlauum computes the product U * U**H or L**H * L, where the triangular
// factor U or L is stored in the upper or lower triangular part of
// the array A.
//
// If UPLO = 'U' or 'u' then the upper triangle of the result is stored,
// overwriting the factor U in A.
// If UPLO = 'L' or 'l' then the lower triangle of the result is stored,
// overwriting the factor L in A.
//
// This is the blocked form of the algorithm, calling Level 3 BLAS.
func Zlauum(uplo mat.MatUplo, n int, a *mat.CMatrix) (err error) {
	var upper bool
	var cone complex128
	var one float64
	var i, ib, nb int

	one = 1.0
	cone = (1.0 + 0.0*1i)

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
		gltest.Xerbla2("Zlauum", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Determine the block size for this environment.
	nb = Ilaenv(1, "Zlauum", []byte{uplo.Byte()}, n, -1, -1, -1)

	if nb <= 1 || nb >= n {
		//        Use unblocked code
		if err = Zlauu2(uplo, n, a); err != nil {
			panic(err)
		}
	} else {
		//        Use blocked code
		if upper {
			//           Compute the product U * U**H.
			for i = 1; i <= n; i += nb {
				ib = min(nb, n-i+1)
				if err = a.Off(0, i-1).Trmm(Right, Upper, ConjTrans, NonUnit, i-1, ib, cone, a.Off(i-1, i-1)); err != nil {
					panic(err)
				}
				if err = Zlauu2(Upper, ib, a.Off(i-1, i-1)); err != nil {
					panic(err)
				}
				if i+ib <= n {
					if err = a.Off(0, i-1).Gemm(NoTrans, ConjTrans, i-1, ib, n-i-ib+1, cone, a.Off(0, i+ib-1), a.Off(i-1, i+ib-1), cone); err != nil {
						panic(err)
					}
					if err = a.Off(i-1, i-1).Herk(Upper, NoTrans, ib, n-i-ib+1, one, a.Off(i-1, i+ib-1), one); err != nil {
						panic(err)
					}
				}
			}
		} else {
			//           Compute the product L**H * L.
			for i = 1; i <= n; i += nb {
				ib = min(nb, n-i+1)
				if err = a.Off(i-1, 0).Trmm(Left, Lower, ConjTrans, NonUnit, ib, i-1, cone, a.Off(i-1, i-1)); err != nil {
					panic(err)
				}
				if err = Zlauu2(Lower, ib, a.Off(i-1, i-1)); err != nil {
					panic(err)
				}
				if i+ib <= n {
					if err = a.Off(i-1, 0).Gemm(ConjTrans, NoTrans, ib, i-1, n-i-ib+1, cone, a.Off(i+ib-1, i-1), a.Off(i+ib-1, 0), cone); err != nil {
						panic(err)
					}
					if err = a.Off(i-1, i-1).Herk(Lower, ConjTrans, ib, n-i-ib+1, one, a.Off(i+ib-1, i-1), one); err != nil {
						panic(err)
					}
				}
			}
		}
	}

	return
}
