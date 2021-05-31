package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhpgst reduces a complex Hermitian-definite generalized
// eigenproblem to standard form, using packed storage.
//
// If ITYPE = 1, the problem is A*x = lambda*B*x,
// and A is overwritten by inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H)
//
// If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
// B*A*x = lambda*x, and A is overwritten by U*A*U**H or L**H*A*L.
//
// B must have been previously factorized as U**H*U or L*L**H by ZPPTRF.
func Zhpgst(itype *int, uplo byte, n *int, ap *mat.CVector, bp *mat.CVector, info *int) {
	var upper bool
	var cone, ct complex128
	var ajj, akk, bjj, bkk, half, one float64
	var j, j1, j1j1, jj, k, k1, k1k1, kk int

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
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHPGST"), -(*info))
		return
	}

	if (*itype) == 1 {
		if upper {
			//           Compute inv(U**H)*A*inv(U)
			//
			//           J1 and JJ are the indices of A(1,j) and A(j,j)
			jj = 0
			for j = 1; j <= (*n); j++ {
				j1 = jj + 1
				jj = jj + j

				//              Compute the j-th column of the upper triangle of A
				ap.Set(jj-1, ap.GetReCmplx(jj-1))
				bjj = bp.GetRe(jj - 1)
				goblas.Ztpsv(mat.UploByte(uplo), ConjTrans, NonUnit, &j, bp, ap.Off(j1-1), func() *int { y := 1; return &y }())
				goblas.Zhpmv(mat.UploByte(uplo), toPtr(j-1), toPtrc128(-cone), ap, bp.Off(j1-1), func() *int { y := 1; return &y }(), &cone, ap.Off(j1-1), func() *int { y := 1; return &y }())
				goblas.Zdscal(toPtr(j-1), toPtrf64(one/bjj), ap.Off(j1-1), func() *int { y := 1; return &y }())
				ap.Set(jj-1, (ap.Get(jj-1)-goblas.Zdotc(toPtr(j-1), ap.Off(j1-1), func() *int { y := 1; return &y }(), bp.Off(j1-1), func() *int { y := 1; return &y }()))/complex(bjj, 0))
			}
		} else {
			//           Compute inv(L)*A*inv(L**H)
			//
			//           KK and K1K1 are the indices of A(k,k) and A(k+1,k+1)
			kk = 1
			for k = 1; k <= (*n); k++ {
				k1k1 = kk + (*n) - k + 1

				//              Update the lower triangle of A(k:n,k:n)
				akk = ap.GetRe(kk - 1)
				bkk = bp.GetRe(kk - 1)
				akk = akk / math.Pow(bkk, 2)
				ap.SetRe(kk-1, akk)
				if k < (*n) {
					goblas.Zdscal(toPtr((*n)-k), toPtrf64(one/bkk), ap.Off(kk+1-1), func() *int { y := 1; return &y }())
					ct = complex(-half*akk, 0)
					goblas.Zaxpy(toPtr((*n)-k), &ct, bp.Off(kk+1-1), func() *int { y := 1; return &y }(), ap.Off(kk+1-1), func() *int { y := 1; return &y }())
					goblas.Zhpr2(mat.UploByte(uplo), toPtr((*n)-k), toPtrc128(-cone), ap.Off(kk+1-1), func() *int { y := 1; return &y }(), bp.Off(kk+1-1), func() *int { y := 1; return &y }(), ap.Off(k1k1-1))
					goblas.Zaxpy(toPtr((*n)-k), &ct, bp.Off(kk+1-1), func() *int { y := 1; return &y }(), ap.Off(kk+1-1), func() *int { y := 1; return &y }())
					goblas.Ztpsv(mat.UploByte(uplo), NoTrans, NonUnit, toPtr((*n)-k), bp.Off(k1k1-1), ap.Off(kk+1-1), func() *int { y := 1; return &y }())
				}
				kk = k1k1
			}
		}
	} else {
		if upper {
			//           Compute U*A*U**H
			//
			//           K1 and KK are the indices of A(1,k) and A(k,k)
			kk = 0
			for k = 1; k <= (*n); k++ {
				k1 = kk + 1
				kk = kk + k

				//              Update the upper triangle of A(1:k,1:k)
				akk = ap.GetRe(kk - 1)
				bkk = bp.GetRe(kk - 1)
				goblas.Ztpmv(mat.UploByte(uplo), NoTrans, NonUnit, toPtr(k-1), bp, ap.Off(k1-1), func() *int { y := 1; return &y }())
				ct = complex(half*akk, 0)
				goblas.Zaxpy(toPtr(k-1), &ct, bp.Off(k1-1), func() *int { y := 1; return &y }(), ap.Off(k1-1), func() *int { y := 1; return &y }())
				goblas.Zhpr2(mat.UploByte(uplo), toPtr(k-1), &cone, ap.Off(k1-1), func() *int { y := 1; return &y }(), bp.Off(k1-1), func() *int { y := 1; return &y }(), ap)
				goblas.Zaxpy(toPtr(k-1), &ct, bp.Off(k1-1), func() *int { y := 1; return &y }(), ap.Off(k1-1), func() *int { y := 1; return &y }())
				goblas.Zdscal(toPtr(k-1), &bkk, ap.Off(k1-1), func() *int { y := 1; return &y }())
				ap.SetRe(kk-1, akk*math.Pow(bkk, 2))
			}
		} else {
			//           Compute L**H *A*L
			//
			//           JJ and J1J1 are the indices of A(j,j) and A(j+1,j+1)
			jj = 1
			for j = 1; j <= (*n); j++ {
				j1j1 = jj + (*n) - j + 1

				//              Compute the j-th column of the lower triangle of A
				ajj = ap.GetRe(jj - 1)
				bjj = bp.GetRe(jj - 1)
				ap.Set(jj-1, complex(ajj*bjj, 0)+goblas.Zdotc(toPtr((*n)-j), ap.Off(jj+1-1), func() *int { y := 1; return &y }(), bp.Off(jj+1-1), func() *int { y := 1; return &y }()))
				goblas.Zdscal(toPtr((*n)-j), &bjj, ap.Off(jj+1-1), func() *int { y := 1; return &y }())
				goblas.Zhpmv(mat.UploByte(uplo), toPtr((*n)-j), &cone, ap.Off(j1j1-1), bp.Off(jj+1-1), func() *int { y := 1; return &y }(), &cone, ap.Off(jj+1-1), func() *int { y := 1; return &y }())
				goblas.Ztpmv(mat.UploByte(uplo), ConjTrans, NonUnit, toPtr((*n)-j+1), bp.Off(jj-1), ap.Off(jj-1), func() *int { y := 1; return &y }())
				jj = j1j1
			}
		}
	}
}
