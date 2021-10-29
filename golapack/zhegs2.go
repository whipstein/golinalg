package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhegs2 reduces a complex Hermitian-definite generalized
// eigenproblem to standard form.
//
// If ITYPE = 1, the problem is A*x = lambda*B*x,
// and A is overwritten by inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H)
//
// If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
// B*A*x = lambda*x, and A is overwritten by U*A*U**H or L**H *A*L.
//
// B must have been previously factorized as U**H *U or L*L**H by ZPOTRF.
func Zhegs2(itype int, uplo mat.MatUplo, n int, a, b *mat.CMatrix) (err error) {
	var upper bool
	var cone, ct complex128
	var akk, bkk, half, one float64
	var k int

	one = 1.0
	half = 0.5
	cone = (1.0 + 0.0*1i)

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
		gltest.Xerbla2("Zhegs2", err)
		return
	}

	if itype == 1 {
		if upper {
			//           Compute inv(U**H)*A*inv(U)
			for k = 1; k <= n; k++ {
				//              Update the upper triangle of A(k:n,k:n)
				akk = a.GetRe(k-1, k-1)
				bkk = b.GetRe(k-1, k-1)
				akk = akk / math.Pow(bkk, 2)
				a.SetRe(k-1, k-1, akk)
				if k < n {
					goblas.Zdscal(n-k, one/bkk, a.CVector(k-1, k))
					ct = complex(-half*akk, 0)
					Zlacgv(n-k, a.CVector(k-1, k))
					Zlacgv(n-k, b.CVector(k-1, k))
					goblas.Zaxpy(n-k, ct, b.CVector(k-1, k), a.CVector(k-1, k))
					if err = goblas.Zher2(uplo, n-k, -cone, a.CVector(k-1, k), b.CVector(k-1, k), a.Off(k, k)); err != nil {
						panic(err)
					}
					goblas.Zaxpy(n-k, ct, b.CVector(k-1, k), a.CVector(k-1, k))
					Zlacgv(n-k, b.CVector(k-1, k))
					if err = goblas.Ztrsv(uplo, ConjTrans, NonUnit, n-k, b.Off(k, k), a.CVector(k-1, k)); err != nil {
						panic(err)
					}
					Zlacgv(n-k, a.CVector(k-1, k))
				}
			}
		} else {
			//           Compute inv(L)*A*inv(L**H)
			for k = 1; k <= n; k++ {
				//              Update the lower triangle of A(k:n,k:n)
				akk = a.GetRe(k-1, k-1)
				bkk = b.GetRe(k-1, k-1)
				akk = akk / math.Pow(bkk, 2)
				a.SetRe(k-1, k-1, akk)
				if k < n {
					goblas.Zdscal(n-k, one/bkk, a.CVector(k, k-1, 1))
					ct = complex(-half*akk, 0)
					goblas.Zaxpy(n-k, ct, b.CVector(k, k-1, 1), a.CVector(k, k-1, 1))
					if err = goblas.Zher2(uplo, n-k, -cone, a.CVector(k, k-1, 1), b.CVector(k, k-1, 1), a.Off(k, k)); err != nil {
						panic(err)
					}
					goblas.Zaxpy(n-k, ct, b.CVector(k, k-1, 1), a.CVector(k, k-1, 1))
					if err = goblas.Ztrsv(uplo, NoTrans, NonUnit, n-k, b.Off(k, k), a.CVector(k, k-1, 1)); err != nil {
						panic(err)
					}
				}
			}
		}
	} else {
		if upper {
			//           Compute U*A*U**H
			for k = 1; k <= n; k++ {
				//              Update the upper triangle of A(1:k,1:k)
				akk = a.GetRe(k-1, k-1)
				bkk = b.GetRe(k-1, k-1)
				err = goblas.Ztrmv(uplo, NoTrans, NonUnit, k-1, b, a.CVector(0, k-1, 1))
				ct = complex(half*akk, 0)
				goblas.Zaxpy(k-1, ct, b.CVector(0, k-1, 1), a.CVector(0, k-1, 1))
				if err = goblas.Zher2(uplo, k-1, cone, a.CVector(0, k-1, 1), b.CVector(0, k-1, 1), a); err != nil {
					panic(err)
				}
				goblas.Zaxpy(k-1, ct, b.CVector(0, k-1, 1), a.CVector(0, k-1, 1))
				goblas.Zdscal(k-1, bkk, a.CVector(0, k-1, 1))
				a.SetRe(k-1, k-1, akk*math.Pow(bkk, 2))
			}
		} else {
			//           Compute L**H *A*L
			for k = 1; k <= n; k++ {
				//
				//              Update the lower triangle of A(1:k,1:k)
				//
				akk = a.GetRe(k-1, k-1)
				bkk = b.GetRe(k-1, k-1)
				Zlacgv(k-1, a.CVector(k-1, 0))
				if err = goblas.Ztrmv(uplo, ConjTrans, NonUnit, k-1, b, a.CVector(k-1, 0)); err != nil {
					panic(err)
				}
				ct = complex(half*akk, 0)
				Zlacgv(k-1, b.CVector(k-1, 0))
				goblas.Zaxpy(k-1, ct, b.CVector(k-1, 0), a.CVector(k-1, 0))
				if err = goblas.Zher2(uplo, k-1, cone, a.CVector(k-1, 0), b.CVector(k-1, 0), a); err != nil {
					panic(err)
				}
				goblas.Zaxpy(k-1, ct, b.CVector(k-1, 0), a.CVector(k-1, 0))
				Zlacgv(k-1, b.CVector(k-1, 0))
				goblas.Zdscal(k-1, bkk, a.CVector(k-1, 0))
				Zlacgv(k-1, a.CVector(k-1, 0))
				a.SetRe(k-1, k-1, akk*math.Pow(bkk, 2))
			}
		}
	}

	return
}
