package golapack

import (
	"github.com/whipstein/golinalg/goblas"
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
func Dsygst(itype *int, uplo byte, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb, info *int) {
	var upper bool
	var half, one float64
	var k, kb, nb int

	one = 1.0
	half = 0.5

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if (*itype) < 1 || (*itype) > 3 {
		(*info) = -1
	} else if !upper && uplo != 'L' {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -7
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYGST"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Determine the block size for this environment.
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DSYGST"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))

	if nb <= 1 || nb >= (*n) {
		//        Use unblocked code
		Dsygs2(itype, uplo, n, a, lda, b, ldb, info)
	} else {
		//        Use blocked code
		if (*itype) == 1 {
			if upper {
				//              Compute inv(U**T)*A*inv(U)
				for k = 1; k <= (*n); k += nb {
					kb = minint((*n)-k+1, nb)
					//                 Update the upper triangle of A(k:n,k:n)
					Dsygs2(itype, uplo, &kb, a.Off(k-1, k-1), lda, b.Off(k-1, k-1), ldb, info)
					if k+kb <= (*n) {
						goblas.Dtrsm(mat.Left, mat.UploByte(uplo), mat.Trans, mat.NonUnit, &kb, toPtr((*n)-k-kb+1), &one, b.Off(k-1, k-1), ldb, a.Off(k-1, k+kb-1), lda)
						goblas.Dsymm(mat.Left, mat.UploByte(uplo), &kb, toPtr((*n)-k-kb+1), toPtrf64(half), a.Off(k-1, k-1), lda, b.Off(k-1, k+kb-1), ldb, &one, a.Off(k-1, k+kb-1), lda)
						goblas.Dsyr2k(mat.UploByte(uplo), mat.Trans, toPtr((*n)-k-kb+1), &kb, toPtrf64(one), a.Off(k-1, k+kb-1), lda, b.Off(k-1, k+kb-1), ldb, &one, a.Off(k+kb-1, k+kb-1), lda)
						goblas.Dsymm(mat.Left, mat.UploByte(uplo), &kb, toPtr((*n)-k-kb+1), toPtrf64(half), a.Off(k-1, k-1), lda, b.Off(k-1, k+kb-1), ldb, &one, a.Off(k-1, k+kb-1), lda)
						goblas.Dtrsm(mat.Right, mat.UploByte(uplo), mat.NoTrans, mat.NonUnit, &kb, toPtr((*n)-k-kb+1), &one, b.Off(k+kb-1, k+kb-1), ldb, a.Off(k-1, k+kb-1), lda)
					}
				}
			} else {
				//              Compute inv(L)*A*inv(L**T)
				for k = 1; k <= (*n); k += nb {
					kb = minint((*n)-k+1, nb)
					//                 Update the lower triangle of A(k:n,k:n)
					Dsygs2(itype, uplo, &kb, a.Off(k-1, k-1), lda, b.Off(k-1, k-1), ldb, info)
					if k+kb <= (*n) {
						goblas.Dtrsm(mat.Right, mat.UploByte(uplo), mat.Trans, mat.NonUnit, toPtr((*n)-k-kb+1), &kb, &one, b.Off(k-1, k-1), ldb, a.Off(k+kb-1, k-1), lda)
						goblas.Dsymm(mat.Right, mat.UploByte(uplo), toPtr((*n)-k-kb+1), &kb, toPtrf64(half), a.Off(k-1, k-1), lda, b.Off(k+kb-1, k-1), ldb, &one, a.Off(k+kb-1, k-1), lda)
						goblas.Dsyr2k(mat.UploByte(uplo), mat.NoTrans, toPtr((*n)-k-kb+1), &kb, toPtrf64(one), a.Off(k+kb-1, k-1), lda, b.Off(k+kb-1, k-1), ldb, &one, a.Off(k+kb-1, k+kb-1), lda)
						goblas.Dsymm(mat.Right, mat.UploByte(uplo), toPtr((*n)-k-kb+1), &kb, toPtrf64(half), a.Off(k-1, k-1), lda, b.Off(k+kb-1, k-1), ldb, &one, a.Off(k+kb-1, k-1), lda)
						goblas.Dtrsm(mat.Left, mat.UploByte(uplo), mat.NoTrans, mat.NonUnit, toPtr((*n)-k-kb+1), &kb, &one, b.Off(k+kb-1, k+kb-1), ldb, a.Off(k+kb-1, k-1), lda)
					}
				}
			}
		} else {
			if upper {
				//              Compute U*A*U**T
				for k = 1; k <= (*n); k += nb {
					kb = minint((*n)-k+1, nb)
					//                 Update the upper triangle of A(1:k+kb-1,1:k+kb-1)
					goblas.Dtrmm(mat.Left, mat.UploByte(uplo), mat.NoTrans, mat.NonUnit, toPtr(k-1), &kb, &one, b, ldb, a.Off(0, k-1), lda)
					goblas.Dsymm(mat.Right, mat.UploByte(uplo), toPtr(k-1), &kb, &half, a.Off(k-1, k-1), lda, b.Off(0, k-1), ldb, &one, a.Off(0, k-1), lda)
					goblas.Dsyr2k(mat.UploByte(uplo), mat.NoTrans, toPtr(k-1), &kb, &one, a.Off(0, k-1), lda, b.Off(0, k-1), ldb, &one, a, lda)
					goblas.Dsymm(mat.Right, mat.UploByte(uplo), toPtr(k-1), &kb, &half, a.Off(k-1, k-1), lda, b.Off(0, k-1), ldb, &one, a.Off(0, k-1), lda)
					goblas.Dtrmm(mat.Right, mat.UploByte(uplo), mat.Trans, mat.NonUnit, toPtr(k-1), &kb, &one, b.Off(k-1, k-1), ldb, a.Off(0, k-1), lda)
					Dsygs2(itype, uplo, &kb, a.Off(k-1, k-1), lda, b.Off(k-1, k-1), ldb, info)
				}
			} else {
				//              Compute L**T*A*L
				for k = 1; k <= (*n); k += nb {
					kb = minint((*n)-k+1, nb)
					//                 Update the lower triangle of A(1:k+kb-1,1:k+kb-1)
					goblas.Dtrmm(mat.Right, mat.UploByte(uplo), mat.NoTrans, mat.NonUnit, &kb, toPtr(k-1), &one, b, ldb, a.Off(k-1, 0), lda)
					goblas.Dsymm(mat.Left, mat.UploByte(uplo), &kb, toPtr(k-1), &half, a.Off(k-1, k-1), lda, b.Off(k-1, 0), ldb, &one, a.Off(k-1, 0), lda)
					goblas.Dsyr2k(mat.UploByte(uplo), mat.Trans, toPtr(k-1), &kb, &one, a.Off(k-1, 0), lda, b.Off(k-1, 0), ldb, &one, a, lda)
					goblas.Dsymm(mat.Left, mat.UploByte(uplo), &kb, toPtr(k-1), &half, a.Off(k-1, k-1), lda, b.Off(k-1, 0), ldb, &one, a.Off(k-1, 0), lda)
					goblas.Dtrmm(mat.Left, mat.UploByte(uplo), mat.Trans, mat.NonUnit, &kb, toPtr(k-1), &one, b.Off(k-1, k-1), ldb, a.Off(k-1, 0), lda)
					Dsygs2(itype, uplo, &kb, a.Off(k-1, k-1), lda, b.Off(k-1, k-1), ldb, info)
				}
			}
		}
	}
}
