package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Zlaqps computes a step of QR factorization with column pivoting
// of a complex M-by-N matrix A by using Blas-3.  It tries to factorize
// NB columns from A starting from the row OFFSET+1, and updates all
// of the matrix with Blas-3 xGEMM.
//
// In some cases, due to catastrophic cancellations, it cannot
// factorize NB columns.  Hence, the actual number of factorized
// columns is returned in KB.
//
// Block A(1:OFFSET,1:N) is accordingly pivoted, but not factorized.
func Zlaqps(m, n, offset, nb, kb *int, a *mat.CMatrix, lda *int, jpvt *[]int, tau *mat.CVector, vn1, vn2 *mat.Vector, auxv *mat.CVector, f *mat.CMatrix, ldf *int) {
	var akk, cone, czero complex128
	var one, temp, temp2, tol3z, zero float64
	var itemp, j, k, lastrk, lsticc, pvt, rk int
	var err error
	_ = err

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	lastrk = minint(*m, (*n)+(*offset))
	lsticc = 0
	k = 0
	tol3z = math.Sqrt(Dlamch(Epsilon))

	//     Beginning of while loop.
label10:
	;
	if (k < (*nb)) && (lsticc == 0) {
		k = k + 1
		rk = (*offset) + k

		//        Determine ith pivot column and swap if necessary
		pvt = (k - 1) + goblas.Idamax((*n)-k+1, vn1.Off(k-1), 1)
		if pvt != k {
			goblas.Zswap(*m, a.CVector(0, pvt-1), 1, a.CVector(0, k-1), 1)
			goblas.Zswap(k-1, f.CVector(pvt-1, 0), *ldf, f.CVector(k-1, 0), *ldf)
			itemp = (*jpvt)[pvt-1]
			(*jpvt)[pvt-1] = (*jpvt)[k-1]
			(*jpvt)[k-1] = itemp
			vn1.Set(pvt-1, vn1.Get(k-1))
			vn2.Set(pvt-1, vn2.Get(k-1))
		}

		//        Apply previous Householder reflectors to column K:
		//        A(RK:M,K) := A(RK:M,K) - A(RK:M,1:K-1)*F(K,1:K-1)**H.
		if k > 1 {
			for j = 1; j <= k-1; j++ {
				f.Set(k-1, j-1, f.GetConj(k-1, j-1))
			}
			err = goblas.Zgemv(NoTrans, (*m)-rk+1, k-1, -cone, a.Off(rk-1, 0), *lda, f.CVector(k-1, 0), *ldf, cone, a.CVector(rk-1, k-1), 1)
			for j = 1; j <= k-1; j++ {
				f.Set(k-1, j-1, f.GetConj(k-1, j-1))
			}
		}

		//        Generate elementary reflector H(k).
		if rk < (*m) {
			Zlarfg(toPtr((*m)-rk+1), a.GetPtr(rk-1, k-1), a.CVector(rk+1-1, k-1), func() *int { y := 1; return &y }(), tau.GetPtr(k-1))
		} else {
			Zlarfg(func() *int { y := 1; return &y }(), a.GetPtr(rk-1, k-1), a.CVector(rk-1, k-1), func() *int { y := 1; return &y }(), tau.GetPtr(k-1))
		}

		akk = a.Get(rk-1, k-1)
		a.Set(rk-1, k-1, cone)

		//        Compute Kth column of F:
		//
		//        Compute  F(K+1:N,K) := tau(K)*A(RK:M,K+1:N)**H*A(RK:M,K).
		if k < (*n) {
			err = goblas.Zgemv(ConjTrans, (*m)-rk+1, (*n)-k, tau.Get(k-1), a.Off(rk-1, k+1-1), *lda, a.CVector(rk-1, k-1), 1, czero, f.CVector(k+1-1, k-1), 1)
		}

		//        Padding F(1:K,K) with zeros.
		for j = 1; j <= k; j++ {
			f.Set(j-1, k-1, czero)
		}

		//        Incremental updating of F:
		//        F(1:N,K) := F(1:N,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)**H
		//                    *A(RK:M,K).
		if k > 1 {
			err = goblas.Zgemv(ConjTrans, (*m)-rk+1, k-1, -tau.Get(k-1), a.Off(rk-1, 0), *lda, a.CVector(rk-1, k-1), 1, czero, auxv, 1)

			err = goblas.Zgemv(NoTrans, *n, k-1, cone, f, *ldf, auxv.Off(0), 1, cone, f.CVector(0, k-1), 1)
		}

		//        Update the current row of A:
		//        A(RK,K+1:N) := A(RK,K+1:N) - A(RK,1:K)*F(K+1:N,1:K)**H.
		if k < (*n) {
			err = goblas.Zgemm(NoTrans, ConjTrans, 1, (*n)-k, k, -cone, a.Off(rk-1, 0), *lda, f.Off(k+1-1, 0), *ldf, cone, a.Off(rk-1, k+1-1), *lda)
		}

		//        Update partial column norms.
		if rk < lastrk {
			for j = k + 1; j <= (*n); j++ {
				if vn1.Get(j-1) != zero {
					//                 NOTE: The following 4 lines follow from the analysis in
					//                 Lapack Working Note 176.
					temp = a.GetMag(rk-1, j-1) / vn1.Get(j-1)
					temp = maxf64(zero, (one+temp)*(one-temp))
					temp2 = temp * math.Pow(vn1.Get(j-1)/vn2.Get(j-1), 2)
					if temp2 <= tol3z {
						vn2.Set(j-1, float64(lsticc))
						lsticc = j
					} else {
						vn1.Set(j-1, vn1.Get(j-1)*math.Sqrt(temp))
					}
				}
			}
		}

		a.Set(rk-1, k-1, akk)

		//        End of while loop.
		goto label10
	}
	(*kb) = k
	rk = (*offset) + (*kb)

	//     Apply the block reflector to the rest of the matrix:
	//     A(OFFSET+KB+1:M,KB+1:N) := A(OFFSET+KB+1:M,KB+1:N) -
	//                         A(OFFSET+KB+1:M,1:KB)*F(KB+1:N,1:KB)**H.
	if (*kb) < minint(*n, (*m)-(*offset)) {
		err = goblas.Zgemm(NoTrans, ConjTrans, (*m)-rk, (*n)-(*kb), *kb, -cone, a.Off(rk+1-1, 0), *lda, f.Off((*kb)+1-1, 0), *ldf, cone, a.Off(rk+1-1, (*kb)+1-1), *lda)
	}

	//     Recomputation of difficult columns.
label60:
	;
	if lsticc > 0 {
		itemp = int(math.Round(vn2.Get(lsticc - 1)))
		vn1.Set(lsticc-1, goblas.Dznrm2((*m)-rk, a.CVector(rk+1-1, lsticc-1), 1))

		//        NOTE: The computation of VN1( LSTICC ) relies on the fact that
		//        SNRM2 does not fail on vectors with norm below the value of
		//        SQRT(DLAMCH('S'))
		vn2.Set(lsticc-1, vn1.Get(lsticc-1))
		lsticc = itemp
		goto label60
	}
}
