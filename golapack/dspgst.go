package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
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
	var err error
	_ = err

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
				err = goblas.Dtpsv(mat.UploByte(uplo), Trans, NonUnit, j, bp, ap.Off(j1-1, 1))
				err = goblas.Dspmv(mat.UploByte(uplo), j-1, -one, ap, bp.Off(j1-1, 1), one, ap.Off(j1-1, 1))
				goblas.Dscal(j-1, one/bjj, ap.Off(j1-1, 1))
				ap.Set(jj-1, (ap.Get(jj-1)-goblas.Ddot(j-1, ap.Off(j1-1, 1), bp.Off(j1-1, 1)))/bjj)
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
					goblas.Dscal((*n)-k, one/bkk, ap.Off(kk, 1))
					ct = -half * akk
					goblas.Daxpy((*n)-k, ct, bp.Off(kk, 1), ap.Off(kk, 1))
					err = goblas.Dspr2(mat.UploByte(uplo), (*n)-k, -one, ap.Off(kk, 1), bp.Off(kk, 1), ap.Off(k1k1-1))
					goblas.Daxpy((*n)-k, ct, bp.Off(kk, 1), ap.Off(kk, 1))
					err = goblas.Dtpsv(mat.UploByte(uplo), NoTrans, NonUnit, (*n)-k, bp.Off(k1k1-1), ap.Off(kk, 1))
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
				err = goblas.Dtpmv(mat.UploByte(uplo), NoTrans, NonUnit, k-1, bp, ap.Off(k1-1, 1))
				ct = half * akk
				goblas.Daxpy(k-1, ct, bp.Off(k1-1, 1), ap.Off(k1-1, 1))
				err = goblas.Dspr2(mat.UploByte(uplo), k-1, one, ap.Off(k1-1, 1), bp.Off(k1-1, 1), ap)
				goblas.Daxpy(k-1, ct, bp.Off(k1-1, 1), ap.Off(k1-1, 1))
				goblas.Dscal(k-1, bkk, ap.Off(k1-1, 1))
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
				ap.Set(jj-1, ajj*bjj+goblas.Ddot((*n)-j, ap.Off(jj, 1), bp.Off(jj, 1)))
				goblas.Dscal((*n)-j, bjj, ap.Off(jj, 1))
				err = goblas.Dspmv(mat.UploByte(uplo), (*n)-j, one, ap.Off(j1j1-1), bp.Off(jj, 1), one, ap.Off(jj, 1))
				err = goblas.Dtpmv(mat.UploByte(uplo), Trans, NonUnit, (*n)-j+1, bp.Off(jj-1), ap.Off(jj-1, 1))
				jj = j1j1
			}
		}
	}
}
