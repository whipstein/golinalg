package golapack

import (
	"golinalg/mat"
	"math"
	"math/cmplx"
)

// Ztgex2 swaps adjacent diagonal 1 by 1 blocks (A11,B11) and (A22,B22)
// in an upper triangular matrix pair (A, B) by an unitary equivalence
// transformation.
//
// (A, B) must be in generalized Schur canonical form, that is, A and
// B are both upper triangular.
//
// Optionally, the matrices Q and Z of generalized Schur vectors are
// updated.
//
//        Q(in) * A(in) * Z(in)**H = Q(out) * A(out) * Z(out)**H
//        Q(in) * B(in) * Z(in)**H = Q(out) * B(out) * Z(out)**H
func Ztgex2(wantq, wantz bool, n *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, q *mat.CMatrix, ldq *int, z *mat.CMatrix, ldz, j1, info *int) {
	var dtrong, wands, weak bool
	var cdum, cone, czero, f, g, sq, sz complex128
	var cq, cz, eps, sa, sb, scale, smlnum, ss, sum, thresh, twenty, ws float64
	var i, ldst, m int
	work := cvf(8)
	s := cmf(2, 2, opts)
	t := cmf(2, 2, opts)

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	twenty = 2.0e+1
	ldst = 2
	wands = true

	(*info) = 0

	//     Quick return if possible
	if (*n) <= 1 {
		return
	}

	m = ldst
	weak = false
	dtrong = false

	//     Make a local copy of selected block in (A, B)
	Zlacpy('F', &m, &m, a.Off((*j1)-1, (*j1)-1), lda, s, &ldst)
	Zlacpy('F', &m, &m, b.Off((*j1)-1, (*j1)-1), ldb, t, &ldst)

	//     Compute the threshold for testing the acceptance of swapping.
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum) / eps
	scale = real(czero)
	sum = real(cone)
	Zlacpy('F', &m, &m, s, &ldst, work.CMatrix(m, opts), &m)
	Zlacpy('F', &m, &m, t, &ldst, work.CMatrixOff(m*m+1-1, m, opts), &m)
	Zlassq(toPtr(2*m*m), work, func() *int { y := 1; return &y }(), &scale, &sum)
	sa = scale * math.Sqrt(sum)

	//     THRES has been changed from
	//        THRESH = maxint( TEN*EPS*SA, SMLNUM )
	//     to
	//        THRESH = maxint( TWENTY*EPS*SA, SMLNUM )
	//     on 04/01/10.
	//     "Bug" reported by Ondra Kamenik, confirmed by Julie Langou, fixed by
	//     Jim Demmel and Guillaume Revy. See forum post 1783.
	thresh = maxf64(twenty*eps*sa, smlnum)

	//     Compute unitary QL and RQ that swap 1-by-1 and 1-by-1 blocks
	//     using Givens rotations and perform the swap tentatively.
	f = s.Get(1, 1)*t.Get(0, 0) - t.Get(1, 1)*s.Get(0, 0)
	g = s.Get(1, 1)*t.Get(0, 1) - t.Get(1, 1)*s.Get(0, 1)
	sa = s.GetMag(1, 1)
	sb = t.GetMag(1, 1)
	Zlartg(&g, &f, &cz, &sz, &cdum)
	sz = -sz
	Zrot(func() *int { y := 2; return &y }(), s.CVector(0, 0), func() *int { y := 1; return &y }(), s.CVector(0, 1), func() *int { y := 1; return &y }(), &cz, toPtrc128(cmplx.Conj(sz)))
	Zrot(func() *int { y := 2; return &y }(), t.CVector(0, 0), func() *int { y := 1; return &y }(), t.CVector(0, 1), func() *int { y := 1; return &y }(), &cz, toPtrc128(cmplx.Conj(sz)))
	if sa >= sb {
		Zlartg(s.GetPtr(0, 0), s.GetPtr(1, 0), &cq, &sq, &cdum)
	} else {
		Zlartg(t.GetPtr(0, 0), t.GetPtr(1, 0), &cq, &sq, &cdum)
	}
	Zrot(func() *int { y := 2; return &y }(), s.CVector(0, 0), &ldst, s.CVector(1, 0), &ldst, &cq, &sq)
	Zrot(func() *int { y := 2; return &y }(), t.CVector(0, 0), &ldst, t.CVector(1, 0), &ldst, &cq, &sq)

	//     Weak stability test: |S21| + |T21| <= O(EPS F-norm((S, T)))
	ws = s.GetMag(1, 0) + t.GetMag(1, 0)
	weak = ws <= thresh
	if !weak {
		goto label20
	}

	if wands {
		//        Strong stability test:
		//           F-norm((A-QL**H*S*QR, B-QL**H*T*QR)) <= O(EPS*F-norm((A, B)))
		Zlacpy('F', &m, &m, s, &ldst, work.CMatrix(m, opts), &m)
		Zlacpy('F', &m, &m, t, &ldst, work.CMatrixOff(m*m+1-1, m, opts), &m)
		Zrot(func() *int { y := 2; return &y }(), work, func() *int { y := 1; return &y }(), work.Off(2), func() *int { y := 1; return &y }(), &cz, toPtrc128(-cmplx.Conj(sz)))
		Zrot(func() *int { y := 2; return &y }(), work.Off(4), func() *int { y := 1; return &y }(), work.Off(6), func() *int { y := 1; return &y }(), &cz, toPtrc128(-cmplx.Conj(sz)))
		Zrot(func() *int { y := 2; return &y }(), work, func() *int { y := 2; return &y }(), work.Off(1), func() *int { y := 2; return &y }(), &cq, toPtrc128(-sq))
		Zrot(func() *int { y := 2; return &y }(), work.Off(4), func() *int { y := 2; return &y }(), work.Off(5), func() *int { y := 2; return &y }(), &cq, toPtrc128(-sq))
		for i = 1; i <= 2; i++ {
			work.Set(i-1, work.Get(i-1)-a.Get((*j1)+i-1-1, (*j1)-1))
			work.Set(i+2-1, work.Get(i+2-1)-a.Get((*j1)+i-1-1, (*j1)+1-1))
			work.Set(i+4-1, work.Get(i+4-1)-b.Get((*j1)+i-1-1, (*j1)-1))
			work.Set(i+6-1, work.Get(i+6-1)-b.Get((*j1)+i-1-1, (*j1)+1-1))
		}
		scale = real(czero)
		sum = real(cone)
		Zlassq(toPtr(2*m*m), work, func() *int { y := 1; return &y }(), &scale, &sum)
		ss = scale * math.Sqrt(sum)
		dtrong = ss <= thresh
		if !dtrong {
			goto label20
		}
	}

	//     If the swap is accepted ("weakly" and "strongly"), apply the
	//     equivalence transformations to the original matrix pair (A,B)
	Zrot(toPtr((*j1)+1), a.CVector(0, (*j1)-1), func() *int { y := 1; return &y }(), a.CVector(0, (*j1)+1-1), func() *int { y := 1; return &y }(), &cz, toPtrc128(cmplx.Conj(sz)))
	Zrot(toPtr((*j1)+1), b.CVector(0, (*j1)-1), func() *int { y := 1; return &y }(), b.CVector(0, (*j1)+1-1), func() *int { y := 1; return &y }(), &cz, toPtrc128(cmplx.Conj(sz)))
	Zrot(toPtr((*n)-(*j1)+1), a.CVector((*j1)-1, (*j1)-1), lda, a.CVector((*j1)+1-1, (*j1)-1), lda, &cq, &sq)
	Zrot(toPtr((*n)-(*j1)+1), b.CVector((*j1)-1, (*j1)-1), ldb, b.CVector((*j1)+1-1, (*j1)-1), ldb, &cq, &sq)

	//     Set  N1 by N2 (2,1) blocks to 0
	a.Set((*j1)+1-1, (*j1)-1, czero)
	b.Set((*j1)+1-1, (*j1)-1, czero)

	//     Accumulate transformations into Q and Z if requested.
	if wantz {
		Zrot(n, z.CVector(0, (*j1)-1), func() *int { y := 1; return &y }(), z.CVector(0, (*j1)+1-1), func() *int { y := 1; return &y }(), &cz, toPtrc128(cmplx.Conj(sz)))
	}
	if wantq {
		Zrot(n, q.CVector(0, (*j1)-1), func() *int { y := 1; return &y }(), q.CVector(0, (*j1)+1-1), func() *int { y := 1; return &y }(), &cq, toPtrc128(cmplx.Conj(sq)))
	}

	//     Exit with INFO = 0 if swap was successfully performed.
	return

	//     Exit with INFO = 1 if swap was rejected.
label20:
	;
	(*info) = 1
}
