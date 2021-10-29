package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dlatdf uses the LU factorization of the n-by-n matrix Z computed by
// DGETC2 and computes a contribution to the reciprocal Dif-estimate
// by solving Z * x = b for x, and choosing the r.h.s. b such that
// the norm of x is as large as possible. On entry RHS = b holds the
// contribution from earlier solved sub-systems, and on return RHS = x.
//
// The factorization of Z returned by DGETC2 has the form Z = P*L*U*Q,
// where P and Q are permutation matrices. L is lower triangular with
// unit diagonal elements and U is upper triangular.
func Dlatdf(ijob, n int, z *mat.Matrix, rhs *mat.Vector, rdsum, rdscal float64, ipiv, jpiv *[]int) (rdsumOut, rdscalOut float64) {
	var bm, bp, one, pmone, sminu, splus, temp, zero float64
	var i, j, k, maxdim int
	var err error

	maxdim = 8
	zero = 0.0
	one = 1.0
	rdsumOut = rdsum
	rdscalOut = rdscal

	work := vf(4 * maxdim)
	xm := vf(8)
	xp := vf(8)
	iwork := make([]int, 8)

	if ijob != 2 {
		//        Apply permutations IPIV to RHS
		Dlaswp(1, rhs.Matrix(z.Rows, opts), 1, n-1, *ipiv, 1)

		//        Solve for L-part choosing RHS either to +1 or -1.
		pmone = -one

		for j = 1; j <= n-1; j++ {
			bp = rhs.Get(j-1) + one
			bm = rhs.Get(j-1) - one
			splus = one

			//           Look-ahead for L-part RHS(1:N-1) = + or -1, SPLUS and
			//           SMIN computed more efficiently than in BSOLVE [1].
			splus = splus + goblas.Ddot(n-j, z.Vector(j, j-1, 1), z.Vector(j, j-1, 1))
			sminu = goblas.Ddot(n-j, z.Vector(j, j-1, 1), rhs.Off(j, 1))
			splus = splus * rhs.Get(j-1)
			if splus > sminu {
				rhs.Set(j-1, bp)
			} else if sminu > splus {
				rhs.Set(j-1, bm)
			} else {
				//              In this case the updating sums are equal and we can
				//              choose RHS(J) +1 or -1. The first time this happens
				//              we choose -1, thereafter +1. This is a simple way to
				//              get good estimates of matrices like Byers well-known
				//              example (see [1]). (Not done in BSOLVE.)
				rhs.Set(j-1, rhs.Get(j-1)+pmone)
				pmone = one
			}

			//           Compute the remaining r.h.s.
			temp = -rhs.Get(j - 1)
			goblas.Daxpy(n-j, temp, z.Vector(j, j-1, 1), rhs.Off(j, 1))

		}

		//        Solve for U-part, look-ahead for RHS(N) = +-1. This is not done
		//        in BSOLVE and will hopefully give us a better estimate because
		//        any ill-conditioning of the original matrix is transferred to U
		//        and not to L. U(N, N) is an approximation to sigma_min(LU).
		goblas.Dcopy(n-1, rhs.Off(0, 1), xp.Off(0, 1))
		xp.Set(n-1, rhs.Get(n-1)+one)
		rhs.Set(n-1, rhs.Get(n-1)-one)
		splus = zero
		sminu = zero
		for i = n; i >= 1; i-- {
			temp = one / z.Get(i-1, i-1)
			xp.Set(i-1, xp.Get(i-1)*temp)
			rhs.Set(i-1, rhs.Get(i-1)*temp)
			for k = i + 1; k <= n; k++ {
				xp.Set(i-1, xp.Get(i-1)-xp.Get(k-1)*(z.Get(i-1, k-1)*temp))
				rhs.Set(i-1, rhs.Get(i-1)-rhs.Get(k-1)*(z.Get(i-1, k-1)*temp))
			}
			splus = splus + math.Abs(xp.Get(i-1))
			sminu = sminu + math.Abs(rhs.Get(i-1))
		}
		if splus > sminu {
			goblas.Dcopy(n, xp.Off(0, 1), rhs.Off(0, 1))
		}

		//        Apply the permutations JPIV to the computed solution (RHS)
		Dlaswp(1, rhs.Matrix(z.Rows, opts), 1, n-1, *jpiv, -1)

		//        Compute the sum of squares
		rdscalOut, rdsumOut = Dlassq(n, rhs.Off(0, 1), rdscalOut, rdsumOut)

	} else {
		//        IJOB = 2, Compute approximate nullvector XM of Z
		if temp, err = Dgecon('I', n, z, one, work, &iwork); err != nil {
			panic(err)
		}
		goblas.Dcopy(n, work.Off(n, 1), xm.Off(0, 1))

		//        Compute RHS
		Dlaswp(1, xm.Matrix(z.Rows, opts), 1, n-1, *ipiv, -1)
		temp = one / math.Sqrt(goblas.Ddot(n, xm.Off(0, 1), xm.Off(0, 1)))
		goblas.Dscal(n, temp, xm.Off(0, 1))
		goblas.Dcopy(n, xm.Off(0, 1), xp.Off(0, 1))
		goblas.Daxpy(n, one, rhs.Off(0, 1), xp.Off(0, 1))
		goblas.Daxpy(n, -one, xm.Off(0, 1), rhs.Off(0, 1))
		temp = Dgesc2(n, z, rhs, ipiv, jpiv)
		temp = Dgesc2(n, z, xp, ipiv, jpiv)
		if goblas.Dasum(n, xp.Off(0, 1)) > goblas.Dasum(n, rhs.Off(0, 1)) {
			goblas.Dcopy(n, xp.Off(0, 1), rhs.Off(0, 1))
		}

		//        Compute the sum of squares
		rdscalOut, rdsumOut = Dlassq(n, rhs.Off(0, 1), rdscalOut, rdsumOut)

	}

	return
}
