package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dspgst reduces a real symmetric-definite generalized eigenproblem
// to standard form, using packed storage.
//
// If ITYPE = 1, the problem is A*x = lambda*B*x,
// and A is overwritten by inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T)
//
// If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
// B*A*x = lambda*x, and A is overwritten by U*A*U**T or L**T*A*L.
//
// B must have been previously factorized as U**T*U or L*L**T by DPPTRF.
func Dspgst(itype *int, uplo byte, n *int, ap, bp *mat.Vector, info *int) {
	var upper bool
	var ajj, akk, bjj, bkk, ct, half, one float64
	var j, j1, j1j1, jj, k, k1, k1k1, kk int

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
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DSPGST"), -(*info))
		return
	}

	if (*itype) == 1 {
		if upper {
			//           Compute inv(U**T)*A*inv(U)
			//
			//           J1 and JJ are the indices of A(1,j) and A(j,j)
			jj = 0
			for j = 1; j <= (*n); j++ {
				j1 = jj + 1
				jj = jj + j

				//              Compute the j-th column of the upper triangle of A
				bjj = bp.Get(jj - 1)
				goblas.Dtpsv(mat.UploByte(uplo), Trans, NonUnit, &j, bp, ap.Off(j1-1), func() *int { y := 1; return &y }())
				goblas.Dspmv(mat.UploByte(uplo), toPtr(j-1), toPtrf64(-one), ap, bp.Off(j1-1), func() *int { y := 1; return &y }(), &one, ap.Off(j1-1), func() *int { y := 1; return &y }())
				goblas.Dscal(toPtr(j-1), toPtrf64(one/bjj), ap.Off(j1-1), func() *int { y := 1; return &y }())
				ap.Set(jj-1, (ap.Get(jj-1)-goblas.Ddot(toPtr(j-1), ap.Off(j1-1), func() *int { y := 1; return &y }(), bp.Off(j1-1), func() *int { y := 1; return &y }()))/bjj)
			}
		} else {
			//           Compute inv(L)*A*inv(L**T)
			//
			//           KK and K1K1 are the indices of A(k,k) and A(k+1,k+1)
			kk = 1
			for k = 1; k <= (*n); k++ {
				k1k1 = kk + (*n) - k + 1

				//              Update the lower triangle of A(k:n,k:n)
				akk = ap.Get(kk - 1)
				bkk = bp.Get(kk - 1)
				akk = akk / math.Pow(bkk, 2)
				ap.Set(kk-1, akk)
				if k < (*n) {
					goblas.Dscal(toPtr((*n)-k), toPtrf64(one/bkk), ap.Off(kk+1-1), func() *int { y := 1; return &y }())
					ct = -half * akk
					goblas.Daxpy(toPtr((*n)-k), &ct, bp.Off(kk+1-1), func() *int { y := 1; return &y }(), ap.Off(kk+1-1), func() *int { y := 1; return &y }())
					goblas.Dspr2(mat.UploByte(uplo), toPtr((*n)-k), toPtrf64(-one), ap.Off(kk+1-1), func() *int { y := 1; return &y }(), bp.Off(kk+1-1), func() *int { y := 1; return &y }(), ap.Off(k1k1-1))
					goblas.Daxpy(toPtr((*n)-k), &ct, bp.Off(kk+1-1), func() *int { y := 1; return &y }(), ap.Off(kk+1-1), func() *int { y := 1; return &y }())
					goblas.Dtpsv(mat.UploByte(uplo), NoTrans, NonUnit, toPtr((*n)-k), bp.Off(k1k1-1), ap.Off(kk+1-1), func() *int { y := 1; return &y }())
				}
				kk = k1k1
			}
		}
	} else {
		if upper {
			//           Compute U*A*U**T
			//
			//           K1 and KK are the indices of A(1,k) and A(k,k)
			kk = 0
			for k = 1; k <= (*n); k++ {
				k1 = kk + 1
				kk = kk + k

				//              Update the upper triangle of A(1:k,1:k)
				akk = ap.Get(kk - 1)
				bkk = bp.Get(kk - 1)
				goblas.Dtpmv(mat.UploByte(uplo), NoTrans, NonUnit, toPtr(k-1), bp, ap.Off(k1-1), func() *int { y := 1; return &y }())
				ct = half * akk
				goblas.Daxpy(toPtr(k-1), &ct, bp.Off(k1-1), func() *int { y := 1; return &y }(), ap.Off(k1-1), func() *int { y := 1; return &y }())
				goblas.Dspr2(mat.UploByte(uplo), toPtr(k-1), &one, ap.Off(k1-1), func() *int { y := 1; return &y }(), bp.Off(k1-1), func() *int { y := 1; return &y }(), ap)
				goblas.Daxpy(toPtr(k-1), &ct, bp.Off(k1-1), func() *int { y := 1; return &y }(), ap.Off(k1-1), func() *int { y := 1; return &y }())
				goblas.Dscal(toPtr(k-1), &bkk, ap.Off(k1-1), func() *int { y := 1; return &y }())
				ap.Set(kk-1, akk*math.Pow(bkk, 2))
			}
		} else {
			//           Compute L**T *A*L
			//
			//           JJ and J1J1 are the indices of A(j,j) and A(j+1,j+1)
			jj = 1
			for j = 1; j <= (*n); j++ {
				j1j1 = jj + (*n) - j + 1

				//              Compute the j-th column of the lower triangle of A
				ajj = ap.Get(jj - 1)
				bjj = bp.Get(jj - 1)
				ap.Set(jj-1, ajj*bjj+goblas.Ddot(toPtr((*n)-j), ap.Off(jj+1-1), func() *int { y := 1; return &y }(), bp.Off(jj+1-1), func() *int { y := 1; return &y }()))
				goblas.Dscal(toPtr((*n)-j), &bjj, ap.Off(jj+1-1), func() *int { y := 1; return &y }())
				goblas.Dspmv(mat.UploByte(uplo), toPtr((*n)-j), &one, ap.Off(j1j1-1), bp.Off(jj+1-1), func() *int { y := 1; return &y }(), &one, ap.Off(jj+1-1), func() *int { y := 1; return &y }())
				goblas.Dtpmv(mat.UploByte(uplo), Trans, NonUnit, toPtr((*n)-j+1), bp.Off(jj-1), ap.Off(jj-1), func() *int { y := 1; return &y }())
				jj = j1j1
			}
		}
	}
}
