package golapack

import (
	"fmt"
	"math"

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
					a.Off(k-1, k).CVector().Dscal(n-k, one/bkk, a.Rows)
					ct = complex(-half*akk, 0)
					Zlacgv(n-k, a.Off(k-1, k).CVector(), a.Rows)
					Zlacgv(n-k, b.Off(k-1, k).CVector(), b.Rows)
					a.Off(k-1, k).CVector().Axpy(n-k, ct, b.Off(k-1, k).CVector(), b.Rows, a.Rows)
					if err = a.Off(k, k).Her2(uplo, n-k, -cone, a.Off(k-1, k).CVector(), a.Rows, b.Off(k-1, k).CVector(), b.Rows); err != nil {
						panic(err)
					}
					a.Off(k-1, k).CVector().Axpy(n-k, ct, b.Off(k-1, k).CVector(), b.Rows, a.Rows)
					Zlacgv(n-k, b.Off(k-1, k).CVector(), b.Rows)
					if err = a.Off(k-1, k).CVector().Trsv(uplo, ConjTrans, NonUnit, n-k, b.Off(k, k), a.Rows); err != nil {
						panic(err)
					}
					Zlacgv(n-k, a.Off(k-1, k).CVector(), a.Rows)
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
					a.Off(k, k-1).CVector().Dscal(n-k, one/bkk, 1)
					ct = complex(-half*akk, 0)
					a.Off(k, k-1).CVector().Axpy(n-k, ct, b.Off(k, k-1).CVector(), 1, 1)
					if err = a.Off(k, k).Her2(uplo, n-k, -cone, a.Off(k, k-1).CVector(), 1, b.Off(k, k-1).CVector(), 1); err != nil {
						panic(err)
					}
					a.Off(k, k-1).CVector().Axpy(n-k, ct, b.Off(k, k-1).CVector(), 1, 1)
					if err = a.Off(k, k-1).CVector().Trsv(uplo, NoTrans, NonUnit, n-k, b.Off(k, k), 1); err != nil {
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
				err = a.Off(0, k-1).CVector().Trmv(uplo, NoTrans, NonUnit, k-1, b, 1)
				ct = complex(half*akk, 0)
				a.Off(0, k-1).CVector().Axpy(k-1, ct, b.Off(0, k-1).CVector(), 1, 1)
				if err = a.Her2(uplo, k-1, cone, a.Off(0, k-1).CVector(), 1, b.Off(0, k-1).CVector(), 1); err != nil {
					panic(err)
				}
				a.Off(0, k-1).CVector().Axpy(k-1, ct, b.Off(0, k-1).CVector(), 1, 1)
				a.Off(0, k-1).CVector().Dscal(k-1, bkk, 1)
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
				Zlacgv(k-1, a.Off(k-1, 0).CVector(), a.Rows)
				if err = a.Off(k-1, 0).CVector().Trmv(uplo, ConjTrans, NonUnit, k-1, b, a.Rows); err != nil {
					panic(err)
				}
				ct = complex(half*akk, 0)
				Zlacgv(k-1, b.Off(k-1, 0).CVector(), b.Rows)
				a.Off(k-1, 0).CVector().Axpy(k-1, ct, b.Off(k-1, 0).CVector(), b.Rows, a.Rows)
				if err = a.Her2(uplo, k-1, cone, a.Off(k-1, 0).CVector(), a.Rows, b.Off(k-1, 0).CVector(), b.Rows); err != nil {
					panic(err)
				}
				a.Off(k-1, 0).CVector().Axpy(k-1, ct, b.Off(k-1, 0).CVector(), b.Rows, a.Rows)
				Zlacgv(k-1, b.Off(k-1, 0).CVector(), b.Rows)
				a.Off(k-1, 0).CVector().Dscal(k-1, bkk, a.Rows)
				Zlacgv(k-1, a.Off(k-1, 0).CVector(), a.Rows)
				a.SetRe(k-1, k-1, akk*math.Pow(bkk, 2))
			}
		}
	}

	return
}
