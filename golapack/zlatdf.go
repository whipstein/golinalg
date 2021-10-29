package golapack

import (
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Zlatdf computes the contribution to the reciprocal Dif-estimate
// by solving for x in Z * x = b, where b is chosen such that the norm
// of x is as large as possible. It is assumed that LU decomposition
// of Z has been computed by ZGETC2. On entry RHS = f holds the
// contribution from earlier solved sub-systems, and on return RHS = x.
//
// The factorization of Z returned by ZGETC2 has the form
// Z = P * L * U * Q, where P and Q are permutation matrices. L is lower
// triangular with unit diagonal elements and U is upper triangular.
func Zlatdf(ijob, n int, z *mat.CMatrix, rhs *mat.CVector, rdsum, rdscal float64, ipiv, jpiv *[]int) (rdsumOut, rdscalOut float64) {
	var bm, bp, cone, pmone, temp complex128
	var one, sminu, splus, zero float64
	var i, j, k, maxdim int
	var err error

	maxdim = 2
	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)
	rdsumOut = rdsum
	rdscalOut = rdscal

	work := cvf(maxdim)
	xm := cvf(2)
	xp := cvf(2)
	rwork := vf(2)

	if ijob != 2 {
		//        Apply permutations IPIV to RHS
		Zlaswp(1, rhs.CMatrix(z.Rows, opts), 1, n-1, ipiv, 1)

		//        Solve for L-part choosing RHS either to +1 or -1.
		pmone = -cone
		for j = 1; j <= n-1; j++ {
			bp = rhs.Get(j-1) + cone
			bm = rhs.Get(j-1) - cone
			splus = one

			//           Lockahead for L- part RHS(1:N-1) = +-1
			//           SPLUS and SMIN computed more efficiently than in BSOLVE[1].
			splus = splus + real(goblas.Zdotc(n-j, z.CVector(j, j-1, 1), z.CVector(j, j-1, 1)))
			sminu = real(goblas.Zdotc(n-j, z.CVector(j, j-1, 1), rhs.Off(j, 1)))
			splus = splus * rhs.GetRe(j-1)
			if splus > sminu {
				rhs.Set(j-1, bp)
			} else if sminu > splus {
				rhs.Set(j-1, bm)
			} else {
				//              In this case the updating sums are equal and we can
				//              choose RHS(J) +1 or -1. The first time this happens we
				//              choose -1, thereafter +1. This is a simple way to get
				//              good estimates of matrices like Byers well-known example
				//              (see [1]). (Not done in BSOLVE.)
				rhs.Set(j-1, rhs.Get(j-1)+pmone)
				pmone = cone
			}

			//           Compute the remaining r.h.s.
			temp = -rhs.Get(j - 1)
			goblas.Zaxpy(n-j, temp, z.CVector(j, j-1, 1), rhs.Off(j, 1))
		}

		//        Solve for U- part, lockahead for RHS(N) = +-1. This is not done
		//        In BSOLVE and will hopefully give us a better estimate because
		//        any ill-conditioning of the original matrix is transferred to U
		//        and not to L. U(N, N) is an approximation to sigma_min(LU).
		goblas.Zcopy(n-1, rhs.Off(0, 1), work.Off(0, 1))
		work.Set(n-1, rhs.Get(n-1)+cone)
		rhs.Set(n-1, rhs.Get(n-1)-cone)
		splus = zero
		sminu = zero
		for i = n; i >= 1; i-- {
			temp = cone / z.Get(i-1, i-1)
			work.Set(i-1, work.Get(i-1)*temp)
			rhs.Set(i-1, rhs.Get(i-1)*temp)
			for k = i + 1; k <= n; k++ {
				work.Set(i-1, work.Get(i-1)-work.Get(k-1)*(z.Get(i-1, k-1)*temp))
				rhs.Set(i-1, rhs.Get(i-1)-rhs.Get(k-1)*(z.Get(i-1, k-1)*temp))
			}
			splus = splus + work.GetMag(i-1)
			sminu = sminu + rhs.GetMag(i-1)
		}
		if splus > sminu {
			goblas.Zcopy(n, work.Off(0, 1), rhs.Off(0, 1))
		}

		//        Apply the permutations JPIV to the computed solution (RHS)
		Zlaswp(1, rhs.CMatrix(z.Rows, opts), 1, n-1, jpiv, -1)

		//        Compute the sum of squares
		rdscalOut, rdsumOut = Zlassq(n, rhs.Off(0, 1), rdscalOut, rdsumOut)
		return
	}

	//     ENTRY IJOB = 2
	//
	//     Compute approximate nullvector XM of Z
	if _, err = Zgecon('I', n, z, one, work, rwork); err != nil {
		panic(err)
	}
	goblas.Zcopy(n, work.Off(n, 1), xm.Off(0, 1))

	//     Compute RHS
	Zlaswp(1, xm.CMatrix(z.Rows, opts), 1, n-1, ipiv, -1)
	temp = cone / cmplx.Sqrt(goblas.Zdotc(n, xm.Off(0, 1), xm.Off(0, 1)))
	goblas.Zscal(n, temp, xm.Off(0, 1))
	goblas.Zcopy(n, xm.Off(0, 1), xp.Off(0, 1))
	goblas.Zaxpy(n, cone, rhs.Off(0, 1), xp.Off(0, 1))
	goblas.Zaxpy(n, -cone, xm.Off(0, 1), rhs.Off(0, 1))
	_ = Zgesc2(n, z, rhs, ipiv, jpiv)
	_ = Zgesc2(n, z, xp, ipiv, jpiv)
	if goblas.Dzasum(n, xp.Off(0, 1)) > goblas.Dzasum(n, rhs.Off(0, 1)) {
		goblas.Zcopy(n, xp.Off(0, 1), rhs.Off(0, 1))
	}

	//     Compute the sum of squares
	rdscalOut, rdsumOut = Zlassq(n, rhs.Off(0, 1), rdscalOut, rdsumOut)

	return
}
