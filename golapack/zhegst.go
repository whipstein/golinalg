package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhegst reduces a complex Hermitian-definite generalized
// eigenproblem to standard form.
//
// If ITYPE = 1, the problem is A*x = lambda*B*x,
// and A is overwritten by inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H)
//
// If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
// B*A*x = lambda*x, and A is overwritten by U*A*U**H or L**H*A*L.
//
// B must have been previously factorized as U**H*U or L*L**H by ZPOTRF.
func Zhegst(itype *int, uplo byte, n *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb, info *int) {
	var upper bool
	var cone, half complex128
	var one float64
	var k, kb, nb int
	var err error
	_ = err

	one = 1.0
	cone = (1.0 + 0.0*1i)
	half = (0.5 + 0.0*1i)

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
		gltest.Xerbla([]byte("ZHEGST"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Determine the block size for this environment.
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZHEGST"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))

	if nb <= 1 || nb >= (*n) {
		//        Use unblocked code
		Zhegs2(itype, uplo, n, a, lda, b, ldb, info)
	} else {
		//        Use blocked code
		if (*itype) == 1 {
			if upper {
				//              Compute inv(U**H)*A*inv(U)
				for k = 1; k <= (*n); k += nb {
					kb = minint((*n)-k+1, nb)

					//                 Update the upper triangle of A(k:n,k:n)
					Zhegs2(itype, uplo, &kb, a.Off(k-1, k-1), lda, b.Off(k-1, k-1), ldb, info)
					if k+kb <= (*n) {
						err = goblas.Ztrsm(Left, mat.UploByte(uplo), ConjTrans, NonUnit, kb, (*n)-k-kb+1, cone, b.Off(k-1, k-1), *ldb, a.Off(k-1, k+kb-1), *lda)
						err = goblas.Zhemm(Left, mat.UploByte(uplo), kb, (*n)-k-kb+1, -half, a.Off(k-1, k-1), *lda, b.Off(k-1, k+kb-1), *ldb, cone, a.Off(k-1, k+kb-1), *lda)
						err = goblas.Zher2k(mat.UploByte(uplo), ConjTrans, (*n)-k-kb+1, kb, -cone, a.Off(k-1, k+kb-1), *lda, b.Off(k-1, k+kb-1), *ldb, one, a.Off(k+kb-1, k+kb-1), *lda)
						err = goblas.Zhemm(Left, mat.UploByte(uplo), kb, (*n)-k-kb+1, -half, a.Off(k-1, k-1), *lda, b.Off(k-1, k+kb-1), *ldb, cone, a.Off(k-1, k+kb-1), *lda)
						err = goblas.Ztrsm(Right, mat.UploByte(uplo), NoTrans, NonUnit, kb, (*n)-k-kb+1, cone, b.Off(k+kb-1, k+kb-1), *ldb, a.Off(k-1, k+kb-1), *lda)
					}
				}
			} else {
				//              Compute inv(L)*A*inv(L**H)
				for k = 1; k <= (*n); k += nb {
					kb = minint((*n)-k+1, nb)

					//                 Update the lower triangle of A(k:n,k:n)
					Zhegs2(itype, uplo, &kb, a.Off(k-1, k-1), lda, b.Off(k-1, k-1), ldb, info)
					if k+kb <= (*n) {
						err = goblas.Ztrsm(Right, mat.UploByte(uplo), ConjTrans, NonUnit, (*n)-k-kb+1, kb, cone, b.Off(k-1, k-1), *ldb, a.Off(k+kb-1, k-1), *lda)
						err = goblas.Zhemm(Right, mat.UploByte(uplo), (*n)-k-kb+1, kb, -half, a.Off(k-1, k-1), *lda, b.Off(k+kb-1, k-1), *ldb, cone, a.Off(k+kb-1, k-1), *lda)
						err = goblas.Zher2k(mat.UploByte(uplo), NoTrans, (*n)-k-kb+1, kb, -cone, a.Off(k+kb-1, k-1), *lda, b.Off(k+kb-1, k-1), *ldb, one, a.Off(k+kb-1, k+kb-1), *lda)
						err = goblas.Zhemm(Right, mat.UploByte(uplo), (*n)-k-kb+1, kb, -half, a.Off(k-1, k-1), *lda, b.Off(k+kb-1, k-1), *ldb, cone, a.Off(k+kb-1, k-1), *lda)
						err = goblas.Ztrsm(Left, mat.UploByte(uplo), NoTrans, NonUnit, (*n)-k-kb+1, kb, cone, b.Off(k+kb-1, k+kb-1), *ldb, a.Off(k+kb-1, k-1), *lda)
					}
				}
			}
		} else {
			if upper {
				//              Compute U*A*U**H
				for k = 1; k <= (*n); k += nb {
					kb = minint((*n)-k+1, nb)

					//                 Update the upper triangle of A(1:k+kb-1,1:k+kb-1)
					err = goblas.Ztrmm(Left, mat.UploByte(uplo), NoTrans, NonUnit, k-1, kb, cone, b, *ldb, a.Off(0, k-1), *lda)
					err = goblas.Zhemm(Right, mat.UploByte(uplo), k-1, kb, half, a.Off(k-1, k-1), *lda, b.Off(0, k-1), *ldb, cone, a.Off(0, k-1), *lda)
					err = goblas.Zher2k(mat.UploByte(uplo), NoTrans, k-1, kb, cone, a.Off(0, k-1), *lda, b.Off(0, k-1), *ldb, one, a, *lda)
					err = goblas.Zhemm(Right, mat.UploByte(uplo), k-1, kb, half, a.Off(k-1, k-1), *lda, b.Off(0, k-1), *ldb, cone, a.Off(0, k-1), *lda)
					err = goblas.Ztrmm(Right, mat.UploByte(uplo), ConjTrans, NonUnit, k-1, kb, cone, b.Off(k-1, k-1), *ldb, a.Off(0, k-1), *lda)
					Zhegs2(itype, uplo, &kb, a.Off(k-1, k-1), lda, b.Off(k-1, k-1), ldb, info)
				}
			} else {
				//              Compute L**H*A*L
				for k = 1; k <= (*n); k += nb {
					kb = minint((*n)-k+1, nb)

					//                 Update the lower triangle of A(1:k+kb-1,1:k+kb-1)
					err = goblas.Ztrmm(Right, mat.UploByte(uplo), NoTrans, NonUnit, kb, k-1, cone, b, *ldb, a.Off(k-1, 0), *lda)
					err = goblas.Zhemm(Left, mat.UploByte(uplo), kb, k-1, half, a.Off(k-1, k-1), *lda, b.Off(k-1, 0), *ldb, cone, a.Off(k-1, 0), *lda)
					err = goblas.Zher2k(mat.UploByte(uplo), ConjTrans, k-1, kb, cone, a.Off(k-1, 0), *lda, b.Off(k-1, 0), *ldb, one, a, *lda)
					err = goblas.Zhemm(Left, mat.UploByte(uplo), kb, k-1, half, a.Off(k-1, k-1), *lda, b.Off(k-1, 0), *ldb, cone, a.Off(k-1, 0), *lda)
					err = goblas.Ztrmm(Left, mat.UploByte(uplo), ConjTrans, NonUnit, kb, k-1, cone, b.Off(k-1, k-1), *ldb, a.Off(k-1, 0), *lda)
					Zhegs2(itype, uplo, &kb, a.Off(k-1, k-1), lda, b.Off(k-1, k-1), ldb, info)
				}
			}
		}
	}
}
