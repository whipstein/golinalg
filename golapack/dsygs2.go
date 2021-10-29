package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsygs2 reduces a real symmetric-definite generalized eigenproblem
// to standard form.
//
// If ITYPE = 1, the problem is A*x = lambda*B*x,
// and A is overwritten by inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T)
//
// If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
// B*A*x = lambda*x, and A is overwritten by U*A*U**T or L**T *A*L.
//
// B must have been previously factorized as U**T *U or L*L**T by DPOTRF.
func Dsygs2(itype int, uplo mat.MatUplo, n int, a, b *mat.Matrix) (err error) {
	var upper bool
	var akk, bkk, ct, half, one float64
	var k int

	one = 1.0
	half = 0.5

	//     Test the input parameters.
	upper = uplo == Upper
	if itype < 1 || itype > 3 {
		err = fmt.Errorf("itype < 1 || itype > 3: itype=%v", itype)
	} else if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dsygs2", err)
		return
	}

	if itype == 1 {
		if upper {
			//           Compute inv(U**T)*A*inv(U)
			for k = 1; k <= n; k++ {
				//              Update the upper triangle of A(k:n,k:n)
				akk = a.Get(k-1, k-1)
				bkk = b.Get(k-1, k-1)
				akk = akk / math.Pow(bkk, 2)
				a.Set(k-1, k-1, akk)
				if k < n {
					goblas.Dscal(n-k, one/bkk, a.Vector(k-1, k))
					ct = -half * akk
					goblas.Daxpy(n-k, ct, b.Vector(k-1, k), a.Vector(k-1, k))
					if err = goblas.Dsyr2(uplo, n-k, -one, a.Vector(k-1, k), b.Vector(k-1, k), a.Off(k, k)); err != nil {
						panic(err)
					}
					goblas.Daxpy(n-k, ct, b.Vector(k-1, k), a.Vector(k-1, k))
					if err = goblas.Dtrsv(uplo, Trans, NonUnit, n-k, b.Off(k, k), a.Vector(k-1, k)); err != nil {
						panic(err)
					}
				}
			}
		} else {
			//           Compute inv(L)*A*inv(L**T)
			for k = 1; k <= n; k++ {
				//              Update the lower triangle of A(k:n,k:n)
				akk = a.Get(k-1, k-1)
				bkk = b.Get(k-1, k-1)
				akk = akk / math.Pow(bkk, 2)
				a.Set(k-1, k-1, akk)
				if k < n {
					goblas.Dscal(n-k, one/bkk, a.Vector(k, k-1, 1))
					ct = -half * akk
					goblas.Daxpy(n-k, ct, b.Vector(k, k-1, 1), a.Vector(k, k-1, 1))
					if err = goblas.Dsyr2(uplo, n-k, -one, a.Vector(k, k-1, 1), b.Vector(k, k-1, 1), a.Off(k, k)); err != nil {
						panic(err)
					}
					goblas.Daxpy(n-k, ct, b.Vector(k, k-1, 1), a.Vector(k, k-1, 1))
					if err = goblas.Dtrsv(uplo, NoTrans, NonUnit, n-k, b.Off(k, k), a.Vector(k, k-1, 1)); err != nil {
						panic(err)
					}
				}
			}
		}
	} else {
		if upper {
			//           Compute U*A*U**T
			for k = 1; k <= n; k++ {
				//              Update the upper triangle of A(1:k,1:k)
				akk = a.Get(k-1, k-1)
				bkk = b.Get(k-1, k-1)
				if err = goblas.Dtrmv(uplo, NoTrans, NonUnit, k-1, b, a.Vector(0, k-1, 1)); err != nil {
					panic(err)
				}
				ct = half * akk
				goblas.Daxpy(k-1, ct, b.Vector(0, k-1, 1), a.Vector(0, k-1, 1))
				if err = goblas.Dsyr2(uplo, k-1, one, a.Vector(0, k-1, 1), b.Vector(0, k-1, 1), a); err != nil {
					panic(err)
				}
				goblas.Daxpy(k-1, ct, b.Vector(0, k-1, 1), a.Vector(0, k-1, 1))
				goblas.Dscal(k-1, bkk, a.Vector(0, k-1, 1))
				a.Set(k-1, k-1, akk*math.Pow(bkk, 2))
			}
		} else {
			//           Compute L**T *A*L
			for k = 1; k <= n; k++ {
				//              Update the lower triangle of A(1:k,1:k)
				akk = a.Get(k-1, k-1)
				bkk = b.Get(k-1, k-1)
				if err = goblas.Dtrmv(uplo, Trans, NonUnit, k-1, b, a.Vector(k-1, 0)); err != nil {
					panic(err)
				}
				ct = half * akk
				goblas.Daxpy(k-1, ct, b.Vector(k-1, 0), a.Vector(k-1, 0))
				if err = goblas.Dsyr2(uplo, k-1, one, a.Vector(k-1, 0), b.Vector(k-1, 0), a); err != nil {
					panic(err)
				}
				goblas.Daxpy(k-1, ct, b.Vector(k-1, 0), a.Vector(k-1, 0))
				goblas.Dscal(k-1, bkk, a.Vector(k-1, 0))
				a.Set(k-1, k-1, akk*math.Pow(bkk, 2))
			}
		}
	}

	return
}
