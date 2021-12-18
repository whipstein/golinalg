package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztrtri computes the inverse of a complex upper or lower triangular
// matrix A.
//
// This is the Level 3 BLAS version of the algorithm.
func Ztrtri(uplo mat.MatUplo, diag mat.MatDiag, n int, a *mat.CMatrix) (info int, err error) {
	var nounit, upper bool
	var one, zero complex128
	var j, jb, nb, nn int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	upper = uplo == Upper
	nounit = diag == NonUnit
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if !nounit && diag != Unit {
		err = fmt.Errorf("!nounit && diag != Unit: diag=%s", diag)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Ztrtri", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Check for singularity if non-unit.
	if nounit {
		for info = 1; info <= n; info++ {
			if a.Get(info-1, info-1) == zero {
				return
			}
		}
		info = 0
	}

	//     Determine the block size for this environment.
	nb = Ilaenv(1, "Ztrtri", []byte{uplo.Byte(), diag.Byte()}, n, -1, -1, -1)
	if nb <= 1 || nb >= n {
		//        Use unblocked code
		if err = Ztrti2(uplo, diag, n, a); err != nil {
			panic(err)
		}
	} else {
		//        Use blocked code
		if upper {
			//           Compute inverse of upper triangular matrix
			for j = 1; j <= n; j += nb {
				jb = min(nb, n-j+1)

				//              Compute rows 1:j-1 of current block column
				err = a.Off(0, j-1).Trmm(Left, Upper, NoTrans, diag, j-1, jb, one, a)
				err = a.Off(0, j-1).Trsm(Right, Upper, NoTrans, diag, j-1, jb, -one, a.Off(j-1, j-1))

				//              Compute inverse of current diagonal block
				if err = Ztrti2(Upper, diag, jb, a.Off(j-1, j-1)); err != nil {
					panic(err)
				}
			}
		} else {
			//           Compute inverse of lower triangular matrix
			nn = ((n-1)/nb)*nb + 1
			for j = nn; j >= 1; j -= nb {
				jb = min(nb, n-j+1)
				if j+jb <= n {
					//                 Compute rows j+jb:n of current block column
					err = a.Off(j+jb-1, j-1).Trmm(Left, Lower, NoTrans, diag, n-j-jb+1, jb, one, a.Off(j+jb-1, j+jb-1))
					err = a.Off(j+jb-1, j-1).Trsm(Right, Lower, NoTrans, diag, n-j-jb+1, jb, -one, a.Off(j-1, j-1))
				}

				//              Compute inverse of current diagonal block
				if err = Ztrti2(Lower, diag, jb, a.Off(j-1, j-1)); err != nil {
					panic(err)
				}
			}
		}
	}

	return
}
