package golapack

import (
	"fmt"
	"math"

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
func Zhpgst(itype int, uplo mat.MatUplo, n int, ap, bp *mat.CVector) (err error) {
	var upper bool
	var cone, ct complex128
	var ajj, akk, bjj, bkk, half, one float64
	var j, j1, j1j1, jj, k, k1, k1k1, kk int

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
	}
	if err != nil {
		gltest.Xerbla2("Zhpgst", err)
		return
	}

	if itype == 1 {
		if upper {
			//           Compute inv(U**H)*A*inv(U)
			//
			//           J1 and JJ are the indices of A(1,j) and A(j,j)
			jj = 0
			for j = 1; j <= n; j++ {
				j1 = jj + 1
				jj = jj + j

				//              Compute the j-th column of the upper triangle of A
				ap.Set(jj-1, ap.GetReCmplx(jj-1))
				bjj = bp.GetRe(jj - 1)
				if err = ap.Off(j1-1).Tpsv(uplo, ConjTrans, NonUnit, j, bp, 1); err != nil {
					panic(err)
				}
				if err = ap.Off(j1-1).Hpmv(uplo, j-1, -cone, ap, bp.Off(j1-1), 1, cone, 1); err != nil {
					panic(err)
				}
				ap.Off(j1-1).Dscal(j-1, one/bjj, 1)
				ap.Set(jj-1, (ap.Get(jj-1)-bp.Off(j1-1).Dotc(j-1, ap.Off(j1-1), 1, 1))/complex(bjj, 0))
			}
		} else {
			//           Compute inv(L)*A*inv(L**H)
			//
			//           KK and K1K1 are the indices of A(k,k) and A(k+1,k+1)
			kk = 1
			for k = 1; k <= n; k++ {
				k1k1 = kk + n - k + 1

				//              Update the lower triangle of A(k:n,k:n)
				akk = ap.GetRe(kk - 1)
				bkk = bp.GetRe(kk - 1)
				akk = akk / math.Pow(bkk, 2)
				ap.SetRe(kk-1, akk)
				if k < n {
					ap.Off(kk).Dscal(n-k, one/bkk, 1)
					ct = complex(-half*akk, 0)
					ap.Off(kk).Axpy(n-k, ct, bp.Off(kk), 1, 1)
					if err = ap.Off(k1k1-1).Hpr2(uplo, n-k, -cone, ap.Off(kk), 1, bp.Off(kk), 1); err != nil {
						panic(err)
					}
					ap.Off(kk).Axpy(n-k, ct, bp.Off(kk), 1, 1)
					if err = ap.Off(kk).Tpsv(uplo, NoTrans, NonUnit, n-k, bp.Off(k1k1-1), 1); err != nil {
						panic(err)
					}
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
			for k = 1; k <= n; k++ {
				k1 = kk + 1
				kk = kk + k

				//              Update the upper triangle of A(1:k,1:k)
				akk = ap.GetRe(kk - 1)
				bkk = bp.GetRe(kk - 1)
				if err = ap.Off(k1-1).Tpmv(uplo, NoTrans, NonUnit, k-1, bp, 1); err != nil {
					panic(err)
				}
				ct = complex(half*akk, 0)
				ap.Off(k1-1).Axpy(k-1, ct, bp.Off(k1-1), 1, 1)
				if err = ap.Hpr2(uplo, k-1, cone, ap.Off(k1-1), 1, bp.Off(k1-1), 1); err != nil {
					panic(err)
				}
				ap.Off(k1-1).Axpy(k-1, ct, bp.Off(k1-1), 1, 1)
				ap.Off(k1-1).Dscal(k-1, bkk, 1)
				ap.SetRe(kk-1, akk*math.Pow(bkk, 2))
			}
		} else {
			//           Compute L**H *A*L
			//
			//           JJ and J1J1 are the indices of A(j,j) and A(j+1,j+1)
			jj = 1
			for j = 1; j <= n; j++ {
				j1j1 = jj + n - j + 1

				//              Compute the j-th column of the lower triangle of A
				ajj = ap.GetRe(jj - 1)
				bjj = bp.GetRe(jj - 1)
				ap.Set(jj-1, complex(ajj*bjj, 0)+bp.Off(jj).Dotc(n-j, ap.Off(jj), 1, 1))
				ap.Off(jj).Dscal(n-j, bjj, 1)
				if err = ap.Off(jj).Hpmv(uplo, n-j, cone, ap.Off(j1j1-1), bp.Off(jj), 1, cone, 1); err != nil {
					panic(err)
				}
				if err = ap.Off(jj-1).Tpmv(uplo, ConjTrans, NonUnit, n-j+1, bp.Off(jj-1), 1); err != nil {
					panic(err)
				}
				jj = j1j1
			}
		}
	}

	return
}
