package golapack

import (
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
func Zhegs2(itype *int, uplo byte, n *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb, info *int) {
	var upper bool
	var cone, ct complex128
	var akk, bkk, half, one float64
	var k int
	var err error
	_ = err

	one = 1.0
	half = 0.5
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if (*itype) < 1 || (*itype) > 3 {
		(*info) = -1
	} else if !upper && uplo != 'L' {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*lda) < max(1, *n) {
		(*info) = -5
	} else if (*ldb) < max(1, *n) {
		(*info) = -7
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHEGS2"), -(*info))
		return
	}

	if (*itype) == 1 {
		if upper {
			//           Compute inv(U**H)*A*inv(U)
			for k = 1; k <= (*n); k++ {
				//              Update the upper triangle of A(k:n,k:n)
				akk = a.GetRe(k-1, k-1)
				bkk = b.GetRe(k-1, k-1)
				akk = akk / math.Pow(bkk, 2)
				a.SetRe(k-1, k-1, akk)
				if k < (*n) {
					goblas.Zdscal((*n)-k, one/bkk, a.CVector(k-1, k, *lda))
					ct = complex(-half*akk, 0)
					Zlacgv(toPtr((*n)-k), a.CVector(k-1, k), lda)
					Zlacgv(toPtr((*n)-k), b.CVector(k-1, k), ldb)
					goblas.Zaxpy((*n)-k, ct, b.CVector(k-1, k, *ldb), a.CVector(k-1, k, *lda))
					err = goblas.Zher2(mat.UploByte(uplo), (*n)-k, -cone, a.CVector(k-1, k, *lda), b.CVector(k-1, k, *ldb), a.Off(k, k))
					goblas.Zaxpy((*n)-k, ct, b.CVector(k-1, k, *ldb), a.CVector(k-1, k, *lda))
					Zlacgv(toPtr((*n)-k), b.CVector(k-1, k), ldb)
					err = goblas.Ztrsv(mat.UploByte(uplo), ConjTrans, NonUnit, (*n)-k, b.Off(k, k), a.CVector(k-1, k, *lda))
					Zlacgv(toPtr((*n)-k), a.CVector(k-1, k), lda)
				}
			}
		} else {
			//           Compute inv(L)*A*inv(L**H)
			for k = 1; k <= (*n); k++ {
				//              Update the lower triangle of A(k:n,k:n)
				akk = a.GetRe(k-1, k-1)
				bkk = b.GetRe(k-1, k-1)
				akk = akk / math.Pow(bkk, 2)
				a.SetRe(k-1, k-1, akk)
				if k < (*n) {
					goblas.Zdscal((*n)-k, one/bkk, a.CVector(k, k-1, 1))
					ct = complex(-half*akk, 0)
					goblas.Zaxpy((*n)-k, ct, b.CVector(k, k-1, 1), a.CVector(k, k-1, 1))
					err = goblas.Zher2(mat.UploByte(uplo), (*n)-k, -cone, a.CVector(k, k-1, 1), b.CVector(k, k-1, 1), a.Off(k, k))
					goblas.Zaxpy((*n)-k, ct, b.CVector(k, k-1, 1), a.CVector(k, k-1, 1))
					err = goblas.Ztrsv(mat.UploByte(uplo), NoTrans, NonUnit, (*n)-k, b.Off(k, k), a.CVector(k, k-1, 1))
				}
			}
		}
	} else {
		if upper {
			//           Compute U*A*U**H
			for k = 1; k <= (*n); k++ {
				//              Update the upper triangle of A(1:k,1:k)
				akk = a.GetRe(k-1, k-1)
				bkk = b.GetRe(k-1, k-1)
				err = goblas.Ztrmv(mat.UploByte(uplo), NoTrans, NonUnit, k-1, b, a.CVector(0, k-1, 1))
				ct = complex(half*akk, 0)
				goblas.Zaxpy(k-1, ct, b.CVector(0, k-1, 1), a.CVector(0, k-1, 1))
				err = goblas.Zher2(mat.UploByte(uplo), k-1, cone, a.CVector(0, k-1, 1), b.CVector(0, k-1, 1), a)
				goblas.Zaxpy(k-1, ct, b.CVector(0, k-1, 1), a.CVector(0, k-1, 1))
				goblas.Zdscal(k-1, bkk, a.CVector(0, k-1, 1))
				a.SetRe(k-1, k-1, akk*math.Pow(bkk, 2))
			}
		} else {
			//           Compute L**H *A*L
			for k = 1; k <= (*n); k++ {
				//
				//              Update the lower triangle of A(1:k,1:k)
				//
				akk = a.GetRe(k-1, k-1)
				bkk = b.GetRe(k-1, k-1)
				Zlacgv(toPtr(k-1), a.CVector(k-1, 0), lda)
				err = goblas.Ztrmv(mat.UploByte(uplo), ConjTrans, NonUnit, k-1, b, a.CVector(k-1, 0, *lda))
				ct = complex(half*akk, 0)
				Zlacgv(toPtr(k-1), b.CVector(k-1, 0), ldb)
				goblas.Zaxpy(k-1, ct, b.CVector(k-1, 0, *ldb), a.CVector(k-1, 0, *lda))
				err = goblas.Zher2(mat.UploByte(uplo), k-1, cone, a.CVector(k-1, 0, *lda), b.CVector(k-1, 0, *ldb), a)
				goblas.Zaxpy(k-1, ct, b.CVector(k-1, 0, *ldb), a.CVector(k-1, 0, *lda))
				Zlacgv(toPtr(k-1), b.CVector(k-1, 0), ldb)
				goblas.Zdscal(k-1, bkk, a.CVector(k-1, 0, *lda))
				Zlacgv(toPtr(k-1), a.CVector(k-1, 0), lda)
				a.SetRe(k-1, k-1, akk*math.Pow(bkk, 2))
			}
		}
	}
}
