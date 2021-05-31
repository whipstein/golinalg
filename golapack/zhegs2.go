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
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *n) {
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
					goblas.Zdscal(toPtr((*n)-k), toPtrf64(one/bkk), a.CVector(k-1, k+1-1), lda)
					ct = complex(-half*akk, 0)
					Zlacgv(toPtr((*n)-k), a.CVector(k-1, k+1-1), lda)
					Zlacgv(toPtr((*n)-k), b.CVector(k-1, k+1-1), ldb)
					goblas.Zaxpy(toPtr((*n)-k), &ct, b.CVector(k-1, k+1-1), ldb, a.CVector(k-1, k+1-1), lda)
					goblas.Zher2(mat.UploByte(uplo), toPtr((*n)-k), toPtrc128(-cone), a.CVector(k-1, k+1-1), lda, b.CVector(k-1, k+1-1), ldb, a.Off(k+1-1, k+1-1), lda)
					goblas.Zaxpy(toPtr((*n)-k), &ct, b.CVector(k-1, k+1-1), ldb, a.CVector(k-1, k+1-1), lda)
					Zlacgv(toPtr((*n)-k), b.CVector(k-1, k+1-1), ldb)
					goblas.Ztrsv(mat.UploByte(uplo), ConjTrans, NonUnit, toPtr((*n)-k), b.Off(k+1-1, k+1-1), ldb, a.CVector(k-1, k+1-1), lda)
					Zlacgv(toPtr((*n)-k), a.CVector(k-1, k+1-1), lda)
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
					goblas.Zdscal(toPtr((*n)-k), toPtrf64(one/bkk), a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }())
					ct = complex(-half*akk, 0)
					goblas.Zaxpy(toPtr((*n)-k), &ct, b.CVector(k+1-1, k-1), func() *int { y := 1; return &y }(), a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }())
					goblas.Zher2(mat.UploByte(uplo), toPtr((*n)-k), toPtrc128(-cone), a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }(), b.CVector(k+1-1, k-1), func() *int { y := 1; return &y }(), a.Off(k+1-1, k+1-1), lda)
					goblas.Zaxpy(toPtr((*n)-k), &ct, b.CVector(k+1-1, k-1), func() *int { y := 1; return &y }(), a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }())
					goblas.Ztrsv(mat.UploByte(uplo), NoTrans, NonUnit, toPtr((*n)-k), b.Off(k+1-1, k+1-1), ldb, a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }())
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
				goblas.Ztrmv(mat.UploByte(uplo), NoTrans, NonUnit, toPtr(k-1), b, ldb, a.CVector(0, k-1), func() *int { y := 1; return &y }())
				ct = complex(half*akk, 0)
				goblas.Zaxpy(toPtr(k-1), &ct, b.CVector(0, k-1), func() *int { y := 1; return &y }(), a.CVector(0, k-1), func() *int { y := 1; return &y }())
				goblas.Zher2(mat.UploByte(uplo), toPtr(k-1), &cone, a.CVector(0, k-1), func() *int { y := 1; return &y }(), b.CVector(0, k-1), func() *int { y := 1; return &y }(), a, lda)
				goblas.Zaxpy(toPtr(k-1), &ct, b.CVector(0, k-1), func() *int { y := 1; return &y }(), a.CVector(0, k-1), func() *int { y := 1; return &y }())
				goblas.Zdscal(toPtr(k-1), &bkk, a.CVector(0, k-1), func() *int { y := 1; return &y }())
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
				goblas.Ztrmv(mat.UploByte(uplo), ConjTrans, NonUnit, toPtr(k-1), b, ldb, a.CVector(k-1, 0), lda)
				ct = complex(half*akk, 0)
				Zlacgv(toPtr(k-1), b.CVector(k-1, 0), ldb)
				goblas.Zaxpy(toPtr(k-1), &ct, b.CVector(k-1, 0), ldb, a.CVector(k-1, 0), lda)
				goblas.Zher2(mat.UploByte(uplo), toPtr(k-1), &cone, a.CVector(k-1, 0), lda, b.CVector(k-1, 0), ldb, a, lda)
				goblas.Zaxpy(toPtr(k-1), &ct, b.CVector(k-1, 0), ldb, a.CVector(k-1, 0), lda)
				Zlacgv(toPtr(k-1), b.CVector(k-1, 0), ldb)
				goblas.Zdscal(toPtr(k-1), &bkk, a.CVector(k-1, 0), lda)
				Zlacgv(toPtr(k-1), a.CVector(k-1, 0), lda)
				a.SetRe(k-1, k-1, akk*math.Pow(bkk, 2))
			}
		}
	}
}
