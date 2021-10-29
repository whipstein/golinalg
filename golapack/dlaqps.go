package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dlaqps computes a step of QR factorization with column pivoting
// of a real M-by-N matrix A by using Blas-3.  It tries to factorize
// NB columns from A starting from the row OFFSET+1, and updates all
// of the matrix with Blas-3 xGEMM.
//
// In some cases, due to catastrophic cancellations, it cannot
// factorize NB columns.  Hence, the actual number of factorized
// columns is returned in KB.
//
// Block A(1:OFFSET,1:N) is accordingly pivoted, but not factorized.
func Dlaqps(m, n, offset, nb, kb int, a *mat.Matrix, jpvt *[]int, tau, vn1, vn2, auxv *mat.Vector, f *mat.Matrix) {
	var akk, one, temp, temp2, tol3z, zero float64
	var itemp, j, k, lastrk, lsticc, pvt, rk int
	var err error

	zero = 0.0
	one = 1.0

	lastrk = min(m, n+offset)
	lsticc = 0
	k = 0
	tol3z = math.Sqrt(Dlamch(Epsilon))

	//     Beginning of while loop.
label10:
	;
	if (k < nb) && (lsticc == 0) {
		k = k + 1
		rk = offset + k

		//        Determine ith pivot column and swap if necessary
		pvt = (k - 1) + goblas.Idamax(n-k+1, vn1.Off(k-1))
		if pvt != k {
			goblas.Dswap(m, a.Vector(0, pvt-1, 1), a.Vector(0, k-1, 1))
			goblas.Dswap(k-1, f.Vector(pvt-1, 0, f.Rows), f.Vector(k-1, 0, f.Rows))
			itemp = (*jpvt)[pvt-1]
			(*jpvt)[pvt-1] = (*jpvt)[k-1]
			(*jpvt)[k-1] = itemp
			vn1.Set(pvt-1, vn1.Get(k-1))
			vn2.Set(pvt-1, vn2.Get(k-1))
		}

		//        Apply previous Householder reflectors to column K:
		//        A(RK:M,K) := A(RK:M,K) - A(RK:M,1:K-1)*F(K,1:K-1)**T.
		if k > 1 {
			if err = goblas.Dgemv(NoTrans, m-rk+1, k-1, -one, a.Off(rk-1, 0), f.Vector(k-1, 0, f.Rows), one, a.Vector(rk-1, k-1, 1)); err != nil {
				panic(err)
			}
		}

		//        Generate elementary reflector H(k).
		if rk < m {
			*a.GetPtr(rk-1, k-1), *tau.GetPtr(k - 1) = Dlarfg(m-rk+1, a.Get(rk-1, k-1), a.Vector(rk, k-1, 1))
		} else {
			*a.GetPtr(rk-1, k-1), *tau.GetPtr(k - 1) = Dlarfg(1, a.Get(rk-1, k-1), a.Vector(rk-1, k-1, 1))
		}

		akk = a.Get(rk-1, k-1)
		a.Set(rk-1, k-1, one)

		//        Compute Kth column of F:
		//
		//        Compute  F(K+1:N,K) := tau(K)*A(RK:M,K+1:N)**T*A(RK:M,K).
		if k < n {
			if err = goblas.Dgemv(Trans, m-rk+1, n-k, tau.Get(k-1), a.Off(rk-1, k), a.Vector(rk-1, k-1, 1), zero, f.Vector(k, k-1, 1)); err != nil {
				panic(err)
			}
		}

		//        Padding F(1:K,K) with zeros.
		for j = 1; j <= k; j++ {
			f.Set(j-1, k-1, zero)
		}

		//        Incremental updating of F:
		//        F(1:N,K) := F(1:N,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)**T
		//                    *A(RK:M,K).
		if k > 1 {
			if err = goblas.Dgemv(Trans, m-rk+1, k-1, -tau.Get(k-1), a.Off(rk-1, 0), a.Vector(rk-1, k-1, 1), zero, auxv); err != nil {
				panic(err)
			}

			if err = goblas.Dgemv(NoTrans, n, k-1, one, f.Off(0, 0), auxv, one, f.Vector(0, k-1, 1)); err != nil {
				panic(err)
			}
		}

		//        Update the current row of A:
		//        A(RK,K+1:N) := A(RK,K+1:N) - A(RK,1:K)*F(K+1:N,1:K)**T.
		if k < n {
			if err = goblas.Dgemv(NoTrans, n-k, k, -one, f.Off(k, 0), a.Vector(rk-1, 0), one, a.Vector(rk-1, k)); err != nil {
				panic(err)
			}
		}

		//        Update partial column norms.
		if rk < lastrk {
			for j = k + 1; j <= n; j++ {
				if vn1.Get(j-1) != zero {
					//                 NOTE: The following 4 lines follow from the analysis in
					//                 Lapack Working Note 176.
					temp = math.Abs(a.Get(rk-1, j-1)) / vn1.Get(j-1)
					temp = math.Max(zero, (one+temp)*(one-temp))
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
	kb = k
	rk = offset + kb

	//     Apply the block reflector to the rest of the matrix:
	//     A(OFFSET+KB+1:M,KB+1:N) := A(OFFSET+KB+1:M,KB+1:N) -
	//                         A(OFFSET+KB+1:M,1:KB)*F(KB+1:N,1:KB)**T.
	if kb < min(n, m-offset) {
		if err = goblas.Dgemm(NoTrans, Trans, m-rk, n-kb, kb, -one, a.Off(rk, 0), f.Off(kb, 0), one, a.Off(rk, kb)); err != nil {
			panic(err)
		}
	}

	//     Recomputation of difficult columns.
label40:
	;
	if lsticc > 0 {
		itemp = int(math.Round(vn2.Get(lsticc - 1)))
		vn1.Set(lsticc-1, goblas.Dnrm2(m-rk, a.Vector(rk, lsticc-1, 1)))

		//        NOTE: The computation of VN1( LSTICC ) relies on the fact that
		//        SNRM2 does not fail on vectors with norm below the value of
		//        SQRT(DLAMCH('S'))
		vn2.Set(lsticc-1, vn1.Get(lsticc-1))
		lsticc = itemp
		goto label40
	}
}
