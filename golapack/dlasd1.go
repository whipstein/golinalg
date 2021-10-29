package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlasd1 computes the SVD of an upper bidiagonal N-by-M matrix B,
// where N = NL + NR + 1 and M = N + SQRE. Dlasd1 is called from DLASD0.
//
// A related subroutine DLASD7 handles the case in which the singular
// values (and the singular vectors in factored form) are desired.
//
// Dlasd1 computes the SVD as follows:
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
func Dlasd1(nl, nr, sqre int, d *mat.Vector, alpha, beta float64, u, vt *mat.Matrix, idxq, iwork *[]int, work *mat.Vector) (alphaOut, betaOut float64, info int, err error) {
	var one, orgnrm, zero float64
	var coltyp, i, idx, idxc, idxp, iq, isigma, iu2, ivt2, iz, k, ldq, ldu2, ldvt2, m, n, n1, n2 int

	one = 1.0
	zero = 0.0
	alphaOut = alpha
	betaOut = beta

	//     Test the input parameters.
	if nl < 1 {
		err = fmt.Errorf("nl < 1: nl=%v", nl)
	} else if nr < 1 {
		err = fmt.Errorf("nr < 1: nr=%v", nr)
	} else if (sqre < 0) || (sqre > 1) {
		err = fmt.Errorf("(sqre < 0) || (sqre > 1): sqre=%v", sqre)
	}
	if err != nil {
		gltest.Xerbla2("Dlasd1", err)
		return
	}

	n = nl + nr + 1
	m = n + sqre

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
	orgnrm = math.Max(math.Abs(alphaOut), math.Abs(betaOut))
	d.Set(nl, zero)
	for i = 1; i <= n; i++ {
		if math.Abs(d.Get(i-1)) > orgnrm {
			orgnrm = math.Abs(d.Get(i - 1))
		}
	}
	if err = Dlascl('G', 0, 0, orgnrm, one, n, 1, d.Matrix(n, opts)); err != nil {
		panic(err)
	}
	alphaOut = alphaOut / orgnrm
	betaOut = betaOut / orgnrm

	//     Deflate singular values.
	k, err = Dlasd2(nl, nr, sqre, d, work.Off(iz-1), alphaOut, betaOut, u, vt, work.Off(isigma-1), work.MatrixOff(iu2-1, ldu2, opts), work.MatrixOff(ivt2-1, ldvt2, opts), toSlice(iwork, idxp-1), toSlice(iwork, idx-1), toSlice(iwork, idxc-1), idxq, toSlice(iwork, coltyp-1))

	//     Solve Secular Equation and update singular vectors.
	ldq = k
	info, err = Dlasd3(nl, nr, sqre, k, d, work.MatrixOff(iq-1, ldq, opts), work.Off(isigma-1), u, work.MatrixOff(iu2-1, ldu2, opts), vt, work.MatrixOff(ivt2-1, ldvt2, opts), toSlice(iwork, idxc-1), toSlice(iwork, coltyp-1), work.Off(iz-1))

	//     Report the convergence failure.
	if info != 0 {
		return
	}

	//     Unscale.
	if err = Dlascl('G', 0, 0, one, orgnrm, n, 1, d.Matrix(n, opts)); err != nil {
		panic(err)
	}

	//     Prepare the IDXQ sorting permutation.
	n1 = k
	n2 = n - k
	Dlamrg(n1, n2, d, 1, -1, idxq)

	return
}
