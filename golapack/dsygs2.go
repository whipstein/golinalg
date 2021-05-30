package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
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
func Dsygs2(itype *int, uplo byte, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb, info *int) {
	var upper bool
	var akk, bkk, ct, half, one float64
	var k int

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
		gltest.Xerbla([]byte("DSYGS2"), -(*info))
		return
	}

	if (*itype) == 1 {
		if upper {
			//           Compute inv(U**T)*A*inv(U)
			for k = 1; k <= (*n); k++ {
				//              Update the upper triangle of A(k:n,k:n)
				akk = a.Get(k-1, k-1)
				bkk = b.Get(k-1, k-1)
				akk = akk / math.Pow(bkk, 2)
				a.Set(k-1, k-1, akk)
				if k < (*n) {
					goblas.Dscal(toPtr((*n)-k), toPtrf64(one/bkk), a.Vector(k-1, k+1-1), lda)
					ct = -half * akk
					goblas.Daxpy(toPtr((*n)-k), &ct, b.Vector(k-1, k+1-1), ldb, a.Vector(k-1, k+1-1), lda)
					goblas.Dsyr2(mat.UploByte(uplo), toPtr((*n)-k), toPtrf64(-one), a.Vector(k-1, k+1-1), lda, b.Vector(k-1, k+1-1), ldb, a.Off(k+1-1, k+1-1), lda)
					goblas.Daxpy(toPtr((*n)-k), &ct, b.Vector(k-1, k+1-1), ldb, a.Vector(k-1, k+1-1), lda)
					goblas.Dtrsv(mat.UploByte(uplo), mat.Trans, mat.NonUnit, toPtr((*n)-k), b.Off(k+1-1, k+1-1), ldb, a.Vector(k-1, k+1-1), lda)
				}
			}
		} else {
			//           Compute inv(L)*A*inv(L**T)
			for k = 1; k <= (*n); k++ {
				//              Update the lower triangle of A(k:n,k:n)
				akk = a.Get(k-1, k-1)
				bkk = b.Get(k-1, k-1)
				akk = akk / math.Pow(bkk, 2)
				a.Set(k-1, k-1, akk)
				if k < (*n) {
					goblas.Dscal(toPtr((*n)-k), toPtrf64(one/bkk), a.Vector(k+1-1, k-1), toPtr(1))
					ct = -half * akk
					goblas.Daxpy(toPtr((*n)-k), &ct, b.Vector(k+1-1, k-1), toPtr(1), a.Vector(k+1-1, k-1), toPtr(1))
					goblas.Dsyr2(mat.UploByte(uplo), toPtr((*n)-k), toPtrf64(-one), a.Vector(k+1-1, k-1), toPtr(1), b.Vector(k+1-1, k-1), toPtr(1), a.Off(k+1-1, k+1-1), lda)
					goblas.Daxpy(toPtr((*n)-k), &ct, b.Vector(k+1-1, k-1), toPtr(1), a.Vector(k+1-1, k-1), toPtr(1))
					goblas.Dtrsv(mat.UploByte(uplo), mat.NoTrans, mat.NonUnit, toPtr((*n)-k), b.Off(k+1-1, k+1-1), ldb, a.Vector(k+1-1, k-1), toPtr(1))
				}
			}
		}
	} else {
		if upper {
			//           Compute U*A*U**T
			for k = 1; k <= (*n); k++ {
				//              Update the upper triangle of A(1:k,1:k)
				akk = a.Get(k-1, k-1)
				bkk = b.Get(k-1, k-1)
				goblas.Dtrmv(mat.UploByte(uplo), mat.NoTrans, mat.NonUnit, toPtr(k-1), b, ldb, a.Vector(0, k-1), toPtr(1))
				ct = half * akk
				goblas.Daxpy(toPtr(k-1), &ct, b.Vector(0, k-1), toPtr(1), a.Vector(0, k-1), toPtr(1))
				goblas.Dsyr2(mat.UploByte(uplo), toPtr(k-1), &one, a.Vector(0, k-1), toPtr(1), b.Vector(0, k-1), toPtr(1), a, lda)
				goblas.Daxpy(toPtr(k-1), &ct, b.Vector(0, k-1), toPtr(1), a.Vector(0, k-1), toPtr(1))
				goblas.Dscal(toPtr(k-1), &bkk, a.Vector(0, k-1), toPtr(1))
				a.Set(k-1, k-1, akk*math.Pow(bkk, 2))
			}
		} else {
			//           Compute L**T *A*L
			for k = 1; k <= (*n); k++ {
				//              Update the lower triangle of A(1:k,1:k)
				akk = a.Get(k-1, k-1)
				bkk = b.Get(k-1, k-1)
				goblas.Dtrmv(mat.UploByte(uplo), mat.Trans, mat.NonUnit, toPtr(k-1), b, ldb, a.Vector(k-1, 0), lda)
				ct = half * akk
				goblas.Daxpy(toPtr(k-1), &ct, b.Vector(k-1, 0), ldb, a.Vector(k-1, 0), lda)
				goblas.Dsyr2(mat.UploByte(uplo), toPtr(k-1), &one, a.Vector(k-1, 0), lda, b.Vector(k-1, 0), ldb, a, lda)
				goblas.Daxpy(toPtr(k-1), &ct, b.Vector(k-1, 0), ldb, a.Vector(k-1, 0), lda)
				goblas.Dscal(toPtr(k-1), &bkk, a.Vector(k-1, 0), lda)
				a.Set(k-1, k-1, akk*math.Pow(bkk, 2))
			}
		}
	}
}