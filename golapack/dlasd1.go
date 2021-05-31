package golapack

import (
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlasd1 computes the SVD of an upper bidiagonal N-by-M matrix B,
// where N = NL + NR + 1 and M = N + SQRE. DLASD1 is called from DLASD0.
//
// A related subroutine DLASD7 handles the case in which the singular
// values (and the singular vectors in factored form) are desired.
//
// DLASD1 computes the SVD as follows:
//
//               ( D1(in)    0    0       0 )
//   B = U(in) * (   Z1**T   a   Z2**T    b ) * VT(in)
//               (   0       0   D2(in)   0 )
//
//     = U(out) * ( D(out) 0) * VT(out)
//
// where Z**T = (Z1**T a Z2**T b) = u**T VT**T, and u is a vector of dimension M
// with ALPHA and BETA in the NL+1 and NL+2 th entries and zeros
// elsewhere; and the entry b is empty if SQRE = 0.
//
// The left singular vectors of the original matrix are stored in U, and
// the transpose of the right singular vectors are stored in VT, and the
// singular values are in D.  The algorithm consists of three stages:
//
//    The first stage consists of deflating the size of the problem
//    when there are multiple singular values or when there are zeros in
//    the Z vector.  For each such occurrence the dimension of the
//    secular equation problem is reduced by one.  This stage is
//    performed by the routine DLASD2.
//
//    The second stage consists of calculating the updated
//    singular values. This is done by finding the square roots of the
//    roots of the secular equation via the routine DLASD4 (as called
//    by DLASD3). This routine also calculates the singular vectors of
//    the current problem.
//
//    The final stage consists of computing the updated singular vectors
//    directly using the updated singular values.  The singular vectors
//    for the current problem are multiplied with the singular vectors
//    from the overall problem.
func Dlasd1(nl, nr, sqre *int, d *mat.Vector, alpha, beta *float64, u *mat.Matrix, ldu *int, vt *mat.Matrix, ldvt *int, idxq, iwork *[]int, work *mat.Vector, info *int) {
	var one, orgnrm, zero float64
	var coltyp, i, idx, idxc, idxp, iq, isigma, iu2, ivt2, iz, k, ldq, ldu2, ldvt2, m, n, n1, n2 int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	(*info) = 0

	if (*nl) < 1 {
		(*info) = -1
	} else if (*nr) < 1 {
		(*info) = -2
	} else if ((*sqre) < 0) || ((*sqre) > 1) {
		(*info) = -3
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLASD1"), -(*info))
		return
	}

	n = (*nl) + (*nr) + 1
	m = n + (*sqre)

	//     The following values are for bookkeeping purposes only.  They are
	//     integer pointers which indicate the portion of the workspace
	//     used by a particular array in DLASD2 and DLASD3.
	ldu2 = n
	ldvt2 = m

	iz = 1
	isigma = iz + m
	iu2 = isigma + n
	ivt2 = iu2 + ldu2*n
	iq = ivt2 + ldvt2*m

	idx = 1
	idxc = idx + n
	coltyp = idxc + n
	idxp = coltyp + n

	//     Scale.
	orgnrm = maxf64(math.Abs(*alpha), math.Abs(*beta))
	d.Set((*nl)+1-1, zero)
	for i = 1; i <= n; i++ {
		if math.Abs(d.Get(i-1)) > orgnrm {
			orgnrm = math.Abs(d.Get(i - 1))
		}
	}
	Dlascl('G', toPtr(0), toPtr(0), &orgnrm, &one, &n, toPtr(1), d.Matrix(n, opts), &n, info)
	(*alpha) = (*alpha) / orgnrm
	(*beta) = (*beta) / orgnrm

	//     Deflate singular values.
	Dlasd2(nl, nr, sqre, &k, d, work.Off(iz-1), alpha, beta, u, ldu, vt, ldvt, work.Off(isigma-1), work.MatrixOff(iu2-1, ldu2, opts), &ldu2, work.MatrixOff(ivt2-1, ldvt2, opts), &ldvt2, toSlice(iwork, idxp-1), toSlice(iwork, idx-1), toSlice(iwork, idxc-1), idxq, toSlice(iwork, coltyp-1), info)

	//     Solve Secular Equation and update singular vectors.
	ldq = k
	Dlasd3(nl, nr, sqre, &k, d, work.MatrixOff(iq-1, ldq, opts), &ldq, work.Off(isigma-1), u, ldu, work.MatrixOff(iu2-1, ldu2, opts), &ldu2, vt, ldvt, work.MatrixOff(ivt2-1, ldvt2, opts), &ldvt2, toSlice(iwork, idxc-1), toSlice(iwork, coltyp-1), work.Off(iz-1), info)

	//     Report the convergence failure.
	if (*info) != 0 {
		return
	}

	//     Unscale.
	Dlascl('G', toPtr(0), toPtr(0), &one, &orgnrm, &n, toPtr(1), d.Matrix(n, opts), &n, info)

	//     Prepare the IDXQ sorting permutation.
	n1 = k
	n2 = n - k
	Dlamrg(&n1, &n2, d, toPtr(1), toPtr(-1), idxq)
}
