package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
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
// DLASD6 computes the SVD as follows:
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
// respectively, in DLASD6. Hence U and VT are not explicitly
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
// DLASD6 is called from DLASDA.
func Dlasd6(icompq, nl, nr, sqre *int, d, vf, vl *mat.Vector, alpha, beta *float64, idxq, perm *[]int, givptr *int, givcol *[]int, ldgcol *int, givnum *mat.Matrix, ldgnum *int, poles *mat.Matrix, difl, difr, z *mat.Vector, k *int, c, s *float64, work *mat.Vector, iwork *[]int, info *int) {
	var one, orgnrm, zero float64
	var i, idx, idxc, idxp, isigma, ivfw, ivlw, iw, m, n, n1, n2 int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	(*info) = 0
	n = (*nl) + (*nr) + 1
	m = n + (*sqre)

	if ((*icompq) < 0) || ((*icompq) > 1) {
		(*info) = -1
	} else if (*nl) < 1 {
		(*info) = -2
	} else if (*nr) < 1 {
		(*info) = -3
	} else if ((*sqre) < 0) || ((*sqre) > 1) {
		(*info) = -4
	} else if (*ldgcol) < n {
		(*info) = -14
	} else if (*ldgnum) < n {
		(*info) = -16
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLASD6"), -(*info))
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
	orgnrm = maxf64(math.Abs(*alpha), math.Abs(*beta))
	d.Set((*nl)+1-1, zero)
	for i = 1; i <= n; i++ {
		if math.Abs(d.Get(i-1)) > orgnrm {
			orgnrm = math.Abs(d.Get(i - 1))
		}
	}
	Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &orgnrm, &one, &n, func() *int { y := 1; return &y }(), d.Matrix(n, opts), &n, info)
	(*alpha) = (*alpha) / orgnrm
	(*beta) = (*beta) / orgnrm

	//     Sort and Deflate singular values.
	Dlasd7(icompq, nl, nr, sqre, k, d, z, work.Off(iw-1), vf, work.Off(ivfw-1), vl, work.Off(ivlw-1), alpha, beta, work.Off(isigma-1), toSlice(iwork, idx-1), toSlice(iwork, idxp-1), idxq, perm, givptr, givcol, ldgcol, givnum, ldgnum, c, s, info)

	//     Solve Secular Equation, compute DIFL, DIFR, and update VF, VL.
	Dlasd8(icompq, k, d, z, vf, vl, difl, difr.Matrix(*ldgnum, opts), ldgnum, work.Off(isigma-1), work.Off(iw-1), info)

	//     Report the possible convergence failure.
	if (*info) != 0 {
		return
	}

	//     Save the poles if ICOMPQ = 1.
	if (*icompq) == 1 {
		goblas.Dcopy(k, d, toPtr(1), poles.VectorIdx(0), toPtr(1))
		goblas.Dcopy(k, work.Off(isigma-1), toPtr(1), poles.Vector(0, 1), toPtr(1))
	}

	//     Unscale.
	Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, &orgnrm, &n, func() *int { y := 1; return &y }(), d.Matrix(n, opts), &n, info)

	//     Prepare the IDXQ sorting permutation.
	n1 = (*k)
	n2 = n - (*k)
	Dlamrg(&n1, &n2, d, func() *int { y := 1; return &y }(), toPtr(-1), idxq)
}
