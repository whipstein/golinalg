package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zpotrf computes the Cholesky factorization of a complex Hermitian
// positive definite matrix A.
//
// The factorization has the form
//    A = U**H * U,  if UPLO = 'U', or
//    A = L  * L**H,  if UPLO = 'L',
// where U is an upper triangular matrix and L is lower triangular.
//
// This is the block version of the algorithm, calling Level 3 BLAS.
func Zpotrf(uplo mat.MatUplo, n int, a *mat.CMatrix) (info int, err error) {
	var upper bool
	var cone complex128
	var one float64
	var j, jb, nb int

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
		gltest.Xerbla2("Zpotrf", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Determine the block size for this environment.
	nb = Ilaenv(1, "Zpotrf", []byte{uplo.Byte()}, n, -1, -1, -1)
	if nb <= 1 || nb >= n {
		//        Use unblocked code.
		if info, err = Zpotrf2(uplo, n, a); err != nil {
			panic(err)
		}
	} else {
		//        Use blocked code.
		if upper {
			//           Compute the Cholesky factorization A = U**H *U.
			for j = 1; j <= n; j += nb {
				//              Update and factorize the current diagonal block and test
				//              for non-positive-definiteness.
				jb = min(nb, n-j+1)
				if err = a.Off(j-1, j-1).Herk(Upper, ConjTrans, jb, j-1, -one, a.Off(0, j-1), one); err != nil {
					panic(err)
				}
				if info, err = Zpotrf2(Upper, jb, a.Off(j-1, j-1)); err != nil {
					panic(err)
				}
				if info != 0 {
					goto label30
				}
				if j+jb <= n {
					//                 Compute the current block row.
					if err = a.Off(j-1, j+jb-1).Gemm(ConjTrans, NoTrans, jb, n-j-jb+1, j-1, -cone, a.Off(0, j-1), a.Off(0, j+jb-1), cone); err != nil {
						panic(err)
					}
					if err = a.Off(j-1, j+jb-1).Trsm(Left, Upper, ConjTrans, NonUnit, jb, n-j-jb+1, cone, a.Off(j-1, j-1)); err != nil {
						panic(err)
					}
				}
			}

		} else {
			//           Compute the Cholesky factorization A = L*L**H.
			for j = 1; j <= n; j += nb {
				//              Update and factorize the current diagonal block and test
				//              for non-positive-definiteness.
				jb = min(nb, n-j+1)
				if err = a.Off(j-1, j-1).Herk(Lower, NoTrans, jb, j-1, -one, a.Off(j-1, 0), one); err != nil {
					panic(err)
				}
				if info, err = Zpotrf2(Lower, jb, a.Off(j-1, j-1)); err != nil {
					panic(err)
				}
				if info != 0 {
					goto label30
				}
				if j+jb <= n {
					//                 Compute the current block column.
					if err = a.Off(j+jb-1, j-1).Gemm(NoTrans, ConjTrans, n-j-jb+1, jb, j-1, -cone, a.Off(j+jb-1, 0), a.Off(j-1, 0), cone); err != nil {
						panic(err)
					}
					if err = a.Off(j+jb-1, j-1).Trsm(Right, Lower, ConjTrans, NonUnit, n-j-jb+1, jb, cone, a.Off(j-1, j-1)); err != nil {
						panic(err)
					}
				}
			}
		}
	}
	return

label30:
	;
	info = info + j - 1

	return
}
