package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlasd6 computes the SVD of an updated upper bidiagonal matrix B
// obtained by merging two smaller ones by appending a row. This
// routine is used only for the problem which requires all singular
// values and optionally singular vector matrices in factored form.
// B is an N-by-M matrix with N = NL + NR + 1 and M = N + SQRE.
// A related subroutine, DLASD1, handles the case in which all singular
// values and singular vectors of the bidiagonal matrix are desired.
//
// Dlasd6 computes the SVD as follows:
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
// The singular values of B can be computed using D1, D2, the first
// components of all the right singular vectors of the lower block, and
// the last components of all the right singular vectors of the upper
// block. These components are stored and updated in VF and VL,
// respectively, in Dlasd6. Hence U and VT are not explicitly
// referenced.
//
// The singular values are stored in D. The algorithm consists of two
// stages:
//
//       The first stage consists of deflating the size of the problem
//       when there are multiple singular values or if there is a zero
//       in the Z vector. For each such occurrence the dimension of the
//       secular equation problem is reduced by one. This stage is
//       performed by the routine DLASD7.
//
//       The second stage consists of calculating the updated
//       singular values. This is done by finding the roots of the
//       secular equation via the routine DLASD4 (as called by DLASD8).
//       This routine also updates VF and VL and computes the distances
//       between the updated singular values and the old singular
//       values.
//
// Dlasd6 is called from DLASDA.
func Dlasd6(icompq, nl, nr, sqre int, d, vf, vl *mat.Vector, alpha, beta float64, idxq, perm, givcol *[]int, ldgcol int, givnum, poles *mat.Matrix, difl, difr, z, work *mat.Vector, iwork *[]int) (alphaOut, betaOut float64, givptr, k int, c, s float64, info int, err error) {
	var one, orgnrm, zero float64
	var i, idx, idxc, idxp, isigma, ivfw, ivlw, iw, m, n, n1, n2 int

	one = 1.0
	zero = 0.0
	alphaOut = alpha
	betaOut = beta

	//     Test the input parameters.
	n = nl + nr + 1
	m = n + sqre

	if (icompq < 0) || (icompq > 1) {
		err = fmt.Errorf("(icompq < 0) || (icompq > 1): icompq=%v", icompq)
	} else if nl < 1 {
		err = fmt.Errorf("nl < 1: nl=%v", nl)
	} else if nr < 1 {
		err = fmt.Errorf("nr < 1: nr=%v", nr)
	} else if (sqre < 0) || (sqre > 1) {
		err = fmt.Errorf("(sqre < 0) || (sqre > 1): sqre=%v", sqre)
	} else if ldgcol < n {
		err = fmt.Errorf("ldgcol < n: ldgcol=%v, n=%v", ldgcol, n)
	} else if givnum.Rows < n {
		err = fmt.Errorf("givnum.Rows < n: givnum.Rows=%v, n=%v", givnum.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dlasd6", err)
		return
	}

	//     The following values are for bookkeeping purposes only.  They are
	//     integer pointers which indicate the portion of the workspace
	//     used by a particular array in DLASD7 and DLASD8.
	isigma = 1
	iw = isigma + n
	ivfw = iw + m
	ivlw = ivfw + m

	idx = 1
	idxc = idx + n
	idxp = idxc + n

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

	//     Sort and Deflate singular values.
	if k, givptr, c, s, err = Dlasd7(icompq, nl, nr, sqre, d, z, work.Off(iw-1), vf, work.Off(ivfw-1), vl, work.Off(ivlw-1), alphaOut, betaOut, work.Off(isigma-1), toSlice(iwork, idx-1), toSlice(iwork, idxp-1), idxq, perm, givcol, ldgcol, givnum); err != nil {
		panic(err)
	}

	//     Solve Secular Equation, compute DIFL, DIFR, and update VF, VL.
	if info, err = Dlasd8(icompq, k, d, z, vf, vl, difl, difr.Matrix(givnum.Rows, opts), work.Off(isigma-1), work.Off(iw-1)); err != nil {
		panic(err)
	}

	//     Report the possible convergence failure.
	if info != 0 {
		return
	}

	//     Save the poles if ICOMPQ = 1.
	if icompq == 1 {
		poles.OffIdx(0).Vector().Copy(k, d, 1, 1)
		poles.Off(0, 1).Vector().Copy(k, work.Off(isigma-1), 1, 1)
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
