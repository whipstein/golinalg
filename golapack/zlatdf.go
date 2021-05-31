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
func Zlatdf(ijob, n *int, z *mat.CMatrix, ldz *int, rhs *mat.CVector, rdsum, rdscal *float64, ipiv, jpiv *[]int) {
	var bm, bp, cone, pmone, temp complex128
	var one, rtemp, scale, sminu, splus, zero float64
	var i, info, j, k, maxdim int

	maxdim = 2
	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)
	work := cvf(maxdim)
	xm := cvf(2)
	xp := cvf(2)
	rwork := vf(2)

	if (*ijob) != 2 {
		//        Apply permutations IPIV to RHS
		Zlaswp(func() *int { y := 1; return &y }(), rhs.CMatrix(*ldz, opts), ldz, func() *int { y := 1; return &y }(), toPtr((*n)-1), ipiv, func() *int { y := 1; return &y }())

		//        Solve for L-part choosing RHS either to +1 or -1.
		pmone = -cone
		for j = 1; j <= (*n)-1; j++ {
			bp = rhs.Get(j-1) + cone
			bm = rhs.Get(j-1) - cone
			splus = one

			//           Lockahead for L- part RHS(1:N-1) = +-1
			//           SPLUS and SMIN computed more efficiently than in BSOLVE[1].
			splus = splus + real(goblas.Zdotc(toPtr((*n)-j), z.CVector(j+1-1, j-1), func() *int { y := 1; return &y }(), z.CVector(j+1-1, j-1), func() *int { y := 1; return &y }()))
			sminu = real(goblas.Zdotc(toPtr((*n)-j), z.CVector(j+1-1, j-1), func() *int { y := 1; return &y }(), rhs.Off(j+1-1), func() *int { y := 1; return &y }()))
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
			goblas.Zaxpy(toPtr((*n)-j), &temp, z.CVector(j+1-1, j-1), func() *int { y := 1; return &y }(), rhs.Off(j+1-1), func() *int { y := 1; return &y }())
		}

		//        Solve for U- part, lockahead for RHS(N) = +-1. This is not done
		//        In BSOLVE and will hopefully give us a better estimate because
		//        any ill-conditioning of the original matrix is transferred to U
		//        and not to L. U(N, N) is an approximation to sigma_min(LU).
		goblas.Zcopy(toPtr((*n)-1), rhs, func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
		work.Set((*n)-1, rhs.Get((*n)-1)+cone)
		rhs.Set((*n)-1, rhs.Get((*n)-1)-cone)
		splus = zero
		sminu = zero
		for i = (*n); i >= 1; i-- {
			temp = cone / z.Get(i-1, i-1)
			work.Set(i-1, work.Get(i-1)*temp)
			rhs.Set(i-1, rhs.Get(i-1)*temp)
			for k = i + 1; k <= (*n); k++ {
				work.Set(i-1, work.Get(i-1)-work.Get(k-1)*(z.Get(i-1, k-1)*temp))
				rhs.Set(i-1, rhs.Get(i-1)-rhs.Get(k-1)*(z.Get(i-1, k-1)*temp))
			}
			splus = splus + work.GetMag(i-1)
			sminu = sminu + rhs.GetMag(i-1)
		}
		if splus > sminu {
			goblas.Zcopy(n, work, func() *int { y := 1; return &y }(), rhs, func() *int { y := 1; return &y }())
		}

		//        Apply the permutations JPIV to the computed solution (RHS)
		Zlaswp(func() *int { y := 1; return &y }(), rhs.CMatrix(*ldz, opts), ldz, func() *int { y := 1; return &y }(), toPtr((*n)-1), jpiv, toPtr(-1))

		//        Compute the sum of squares
		Zlassq(n, rhs, func() *int { y := 1; return &y }(), rdscal, rdsum)
		return
	}

	//     ENTRY IJOB = 2
	//
	//     Compute approximate nullvector XM of Z
	Zgecon('I', n, z, ldz, &one, &rtemp, work, rwork, &info)
	goblas.Zcopy(n, work.Off((*n)+1-1), func() *int { y := 1; return &y }(), xm, func() *int { y := 1; return &y }())

	//     Compute RHS
	Zlaswp(func() *int { y := 1; return &y }(), xm.CMatrix(*ldz, opts), ldz, func() *int { y := 1; return &y }(), toPtr((*n)-1), ipiv, toPtr(-1))
	temp = cone / cmplx.Sqrt(goblas.Zdotc(n, xm, func() *int { y := 1; return &y }(), xm, func() *int { y := 1; return &y }()))
	goblas.Zscal(n, &temp, xm, func() *int { y := 1; return &y }())
	goblas.Zcopy(n, xm, func() *int { y := 1; return &y }(), xp, func() *int { y := 1; return &y }())
	goblas.Zaxpy(n, &cone, rhs, func() *int { y := 1; return &y }(), xp, func() *int { y := 1; return &y }())
	goblas.Zaxpy(n, toPtrc128(-cone), xm, func() *int { y := 1; return &y }(), rhs, func() *int { y := 1; return &y }())
	Zgesc2(n, z, ldz, rhs, ipiv, jpiv, &scale)
	Zgesc2(n, z, ldz, xp, ipiv, jpiv, &scale)
	if goblas.Dzasum(n, xp, func() *int { y := 1; return &y }()) > goblas.Dzasum(n, rhs, func() *int { y := 1; return &y }()) {
		goblas.Zcopy(n, xp, func() *int { y := 1; return &y }(), rhs, func() *int { y := 1; return &y }())
	}

	//     Compute the sum of squares
	Zlassq(n, rhs, func() *int { y := 1; return &y }(), rdscal, rdsum)
}
