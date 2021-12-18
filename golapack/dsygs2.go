package golapack

import (
	"fmt"
	"math"

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
					a.Off(k-1, k).Vector().Scal(n-k, one/bkk, a.Rows)
					ct = -half * akk
					a.Off(k-1, k).Vector().Axpy(n-k, ct, b.Off(k-1, k).Vector(), b.Rows, a.Rows)
					if err = a.Off(k, k).Syr2(uplo, n-k, -one, a.Off(k-1, k).Vector(), a.Rows, b.Off(k-1, k).Vector(), b.Rows); err != nil {
						panic(err)
					}
					a.Off(k-1, k).Vector().Axpy(n-k, ct, b.Off(k-1, k).Vector(), b.Rows, a.Rows)
					if err = a.Off(k-1, k).Vector().Trsv(uplo, Trans, NonUnit, n-k, b.Off(k, k), a.Rows); err != nil {
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
					a.Off(k, k-1).Vector().Scal(n-k, one/bkk, 1)
					ct = -half * akk
					a.Off(k, k-1).Vector().Axpy(n-k, ct, b.Off(k, k-1).Vector(), 1, 1)
					if err = a.Off(k, k).Syr2(uplo, n-k, -one, a.Off(k, k-1).Vector(), 1, b.Off(k, k-1).Vector(), 1); err != nil {
						panic(err)
					}
					a.Off(k, k-1).Vector().Axpy(n-k, ct, b.Off(k, k-1).Vector(), 1, 1)
					if err = a.Off(k, k-1).Vector().Trsv(uplo, NoTrans, NonUnit, n-k, b.Off(k, k), 1); err != nil {
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
				if err = a.Off(0, k-1).Vector().Trmv(uplo, NoTrans, NonUnit, k-1, b, 1); err != nil {
					panic(err)
				}
				ct = half * akk
				a.Off(0, k-1).Vector().Axpy(k-1, ct, b.Off(0, k-1).Vector(), 1, 1)
				if err = a.Syr2(uplo, k-1, one, a.Off(0, k-1).Vector(), 1, b.Off(0, k-1).Vector(), 1); err != nil {
					panic(err)
				}
				a.Off(0, k-1).Vector().Axpy(k-1, ct, b.Off(0, k-1).Vector(), 1, 1)
				a.Off(0, k-1).Vector().Scal(k-1, bkk, 1)
				a.Set(k-1, k-1, akk*math.Pow(bkk, 2))
			}
		} else {
			//           Compute L**T *A*L
			for k = 1; k <= n; k++ {
				//              Update the lower triangle of A(1:k,1:k)
				akk = a.Get(k-1, k-1)
				bkk = b.Get(k-1, k-1)
				if err = a.Off(k-1, 0).Vector().Trmv(uplo, Trans, NonUnit, k-1, b, a.Rows); err != nil {
					panic(err)
				}
				ct = half * akk
				a.Off(k-1, 0).Vector().Axpy(k-1, ct, b.Off(k-1, 0).Vector(), b.Rows, a.Rows)
				if err = a.Syr2(uplo, k-1, one, a.Off(k-1, 0).Vector(), a.Rows, b.Off(k-1, 0).Vector(), b.Rows); err != nil {
					panic(err)
				}
				a.Off(k-1, 0).Vector().Axpy(k-1, ct, b.Off(k-1, 0).Vector(), b.Rows, a.Rows)
				a.Off(k-1, 0).Vector().Scal(k-1, bkk, a.Rows)
				a.Set(k-1, k-1, akk*math.Pow(bkk, 2))
			}
		}
	}

	return
}
