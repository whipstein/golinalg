package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsygst reduces a real symmetric-definite generalized eigenproblem
// to standard form.
//
// If ITYPE = 1, the problem is A*x = lambda*B*x,
// and A is overwritten by inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T)
//
// If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
// B*A*x = lambda*x, and A is overwritten by U*A*U**T or L**T*A*L.
//
// B must have been previously factorized as U**T*U or L*L**T by DPOTRF.
func Dsygst(itype int, uplo mat.MatUplo, n int, a, b *mat.Matrix) (err error) {
	var upper bool
	var half, one float64
	var k, kb, nb int

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
		gltest.Xerbla2("Dsygst", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Determine the block size for this environment.
	nb = Ilaenv(1, "Dsygst", []byte{uplo.Byte()}, n, -1, -1, -1)

	if nb <= 1 || nb >= n {
		//        Use unblocked code
		if err = Dsygs2(itype, uplo, n, a, b); err != nil {
			panic(err)
		}
	} else {
		//        Use blocked code
		if itype == 1 {
			if upper {
				//              Compute inv(U**T)*A*inv(U)
				for k = 1; k <= n; k += nb {
					kb = min(n-k+1, nb)
					//                 Update the upper triangle of A(k:n,k:n)
					if err = Dsygs2(itype, uplo, kb, a.Off(k-1, k-1), b.Off(k-1, k-1)); err != nil {
						panic(err)
					}
					if k+kb <= n {
						if err = a.Off(k-1, k+kb-1).Trsm(Left, uplo, Trans, NonUnit, kb, n-k-kb+1, one, b.Off(k-1, k-1)); err != nil {
							panic(err)
						}
						if err = a.Off(k-1, k+kb-1).Symm(Left, uplo, kb, n-k-kb+1, half, a.Off(k-1, k-1), b.Off(k-1, k+kb-1), one); err != nil {
							panic(err)
						}
						if err = a.Off(k+kb-1, k+kb-1).Syr2k(uplo, Trans, n-k-kb+1, kb, one, a.Off(k-1, k+kb-1), b.Off(k-1, k+kb-1), one); err != nil {
							panic(err)
						}
						if err = a.Off(k-1, k+kb-1).Symm(Left, uplo, kb, n-k-kb+1, half, a.Off(k-1, k-1), b.Off(k-1, k+kb-1), one); err != nil {
							panic(err)
						}
						if err = a.Off(k-1, k+kb-1).Trsm(Right, uplo, NoTrans, NonUnit, kb, n-k-kb+1, one, b.Off(k+kb-1, k+kb-1)); err != nil {
							panic(err)
						}
					}
				}
			} else {
				//              Compute inv(L)*A*inv(L**T)
				for k = 1; k <= n; k += nb {
					kb = min(n-k+1, nb)
					//                 Update the lower triangle of A(k:n,k:n)
					if err = Dsygs2(itype, uplo, kb, a.Off(k-1, k-1), b.Off(k-1, k-1)); err != nil {
						panic(err)
					}
					if k+kb <= n {
						if err = a.Off(k+kb-1, k-1).Trsm(Right, uplo, Trans, NonUnit, n-k-kb+1, kb, one, b.Off(k-1, k-1)); err != nil {
							panic(err)
						}
						if err = a.Off(k+kb-1, k-1).Symm(Right, uplo, n-k-kb+1, kb, half, a.Off(k-1, k-1), b.Off(k+kb-1, k-1), one); err != nil {
							panic(err)
						}
						if err = a.Off(k+kb-1, k+kb-1).Syr2k(uplo, NoTrans, n-k-kb+1, kb, one, a.Off(k+kb-1, k-1), b.Off(k+kb-1, k-1), one); err != nil {
							panic(err)
						}
						if err = a.Off(k+kb-1, k-1).Symm(Right, uplo, n-k-kb+1, kb, half, a.Off(k-1, k-1), b.Off(k+kb-1, k-1), one); err != nil {
							panic(err)
						}
						if err = a.Off(k+kb-1, k-1).Trsm(Left, uplo, NoTrans, NonUnit, n-k-kb+1, kb, one, b.Off(k+kb-1, k+kb-1)); err != nil {
							panic(err)
						}
					}
				}
			}
		} else {
			if upper {
				//              Compute U*A*U**T
				for k = 1; k <= n; k += nb {
					kb = min(n-k+1, nb)
					//                 Update the upper triangle of A(1:k+kb-1,1:k+kb-1)
					if err = a.Off(0, k-1).Trmm(Left, uplo, NoTrans, NonUnit, k-1, kb, one, b); err != nil {
						panic(err)
					}
					if err = a.Off(0, k-1).Symm(Right, uplo, k-1, kb, half, a.Off(k-1, k-1), b.Off(0, k-1), one); err != nil {
						panic(err)
					}
					if err = a.Syr2k(uplo, NoTrans, k-1, kb, one, a.Off(0, k-1), b.Off(0, k-1), one); err != nil {
						panic(err)
					}
					if err = a.Off(0, k-1).Symm(Right, uplo, k-1, kb, half, a.Off(k-1, k-1), b.Off(0, k-1), one); err != nil {
						panic(err)
					}
					if err = a.Off(0, k-1).Trmm(Right, uplo, Trans, NonUnit, k-1, kb, one, b.Off(k-1, k-1)); err != nil {
						panic(err)
					}
					if err = Dsygs2(itype, uplo, kb, a.Off(k-1, k-1), b.Off(k-1, k-1)); err != nil {
						panic(err)
					}
				}
			} else {
				//              Compute L**T*A*L
				for k = 1; k <= n; k += nb {
					kb = min(n-k+1, nb)
					//                 Update the lower triangle of A(1:k+kb-1,1:k+kb-1)
					if err = a.Off(k-1, 0).Trmm(Right, uplo, NoTrans, NonUnit, kb, k-1, one, b); err != nil {
						panic(err)
					}
					if err = a.Off(k-1, 0).Symm(Left, uplo, kb, k-1, half, a.Off(k-1, k-1), b.Off(k-1, 0), one); err != nil {
						panic(err)
					}
					if err = a.Syr2k(uplo, Trans, k-1, kb, one, a.Off(k-1, 0), b.Off(k-1, 0), one); err != nil {
						panic(err)
					}
					if err = a.Off(k-1, 0).Symm(Left, uplo, kb, k-1, half, a.Off(k-1, k-1), b.Off(k-1, 0), one); err != nil {
						panic(err)
					}
					if err = a.Off(k-1, 0).Trmm(Left, uplo, Trans, NonUnit, kb, k-1, one, b.Off(k-1, k-1)); err != nil {
						panic(err)
					}
					if err = Dsygs2(itype, uplo, kb, a.Off(k-1, k-1), b.Off(k-1, k-1)); err != nil {
						panic(err)
					}
				}
			}
		}
	}

	return
}
