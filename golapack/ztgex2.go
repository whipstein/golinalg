package golapack

import (
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/mat"
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
func Ztgex2(wantq, wantz bool, n int, a, b, q, z *mat.CMatrix, j1 int) (info int) {
	var dtrong, wands, weak bool
	var cone, czero, f, g, sq, sz complex128
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

	//     Quick return if possible
	if n <= 1 {
		return
	}

	m = ldst
	weak = false
	dtrong = false

	//     Make a local copy of selected block in (A, B)
	Zlacpy(Full, m, m, a.Off(j1-1, j1-1), s)
	Zlacpy(Full, m, m, b.Off(j1-1, j1-1), t)

	//     Compute the threshold for testing the acceptance of swapping.
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum) / eps
	scale = real(czero)
	sum = real(cone)
	Zlacpy(Full, m, m, s, work.CMatrix(m, opts))
	Zlacpy(Full, m, m, t, work.CMatrixOff(m*m, m, opts))
	scale, sum = Zlassq(2*m*m, work.Off(0, 1), scale, sum)
	sa = scale * math.Sqrt(sum)

	//     THRES has been changed from
	//        THRESH = max( TEN*EPS*SA, SMLNUM )
	//     to
	//        THRESH = max( TWENTY*EPS*SA, SMLNUM )
	//     on 04/01/10.
	//     "Bug" reported by Ondra Kamenik, confirmed by Julie Langou, fixed by
	//     Jim Demmel and Guillaume Revy. See forum post 1783.
	thresh = math.Max(twenty*eps*sa, smlnum)

	//     Compute unitary QL and RQ that swap 1-by-1 and 1-by-1 blocks
	//     using Givens rotations and perform the swap tentatively.
	f = s.Get(1, 1)*t.Get(0, 0) - t.Get(1, 1)*s.Get(0, 0)
	g = s.Get(1, 1)*t.Get(0, 1) - t.Get(1, 1)*s.Get(0, 1)
	sa = s.GetMag(1, 1)
	sb = t.GetMag(1, 1)
	cz, sz, _ = Zlartg(g, f)
	sz = -sz
	Zrot(2, s.CVector(0, 0, 1), s.CVector(0, 1, 1), cz, cmplx.Conj(sz))
	Zrot(2, t.CVector(0, 0, 1), t.CVector(0, 1, 1), cz, cmplx.Conj(sz))
	if sa >= sb {
		cq, sq, _ = Zlartg(s.Get(0, 0), s.Get(1, 0))
	} else {
		cq, sq, _ = Zlartg(t.Get(0, 0), t.Get(1, 0))
	}
	Zrot(2, s.CVector(0, 0), s.CVector(1, 0), cq, sq)
	Zrot(2, t.CVector(0, 0), t.CVector(1, 0), cq, sq)

	//     Weak stability test: |S21| + |T21| <= O(EPS F-norm((S, T)))
	ws = s.GetMag(1, 0) + t.GetMag(1, 0)
	weak = ws <= thresh
	if !weak {
		goto label20
	}

	if wands {
		//        Strong stability test:
		//           F-norm((A-QL**H*S*QR, B-QL**H*T*QR)) <= O(EPS*F-norm((A, B)))
		Zlacpy(Full, m, m, s, work.CMatrix(m, opts))
		Zlacpy(Full, m, m, t, work.CMatrixOff(m*m, m, opts))
		Zrot(2, work.Off(0, 1), work.Off(2, 1), cz, -cmplx.Conj(sz))
		Zrot(2, work.Off(4, 1), work.Off(6, 1), cz, -cmplx.Conj(sz))
		Zrot(2, work.Off(0, 2), work.Off(1, 2), cq, -sq)
		Zrot(2, work.Off(4, 2), work.Off(5, 2), cq, -sq)
		for i = 1; i <= 2; i++ {
			work.Set(i-1, work.Get(i-1)-a.Get(j1+i-1-1, j1-1))
			work.Set(i+2-1, work.Get(i+2-1)-a.Get(j1+i-1-1, j1))
			work.Set(i+4-1, work.Get(i+4-1)-b.Get(j1+i-1-1, j1-1))
			work.Set(i+6-1, work.Get(i+6-1)-b.Get(j1+i-1-1, j1))
		}
		scale = real(czero)
		sum = real(cone)
		scale, sum = Zlassq(2*m*m, work.Off(0, 1), scale, sum)
		ss = scale * math.Sqrt(sum)
		dtrong = ss <= thresh
		if !dtrong {
			goto label20
		}
	}

	//     If the swap is accepted ("weakly" and "strongly"), apply the
	//     equivalence transformations to the original matrix pair (A,B)
	Zrot(j1+1, a.CVector(0, j1-1, 1), a.CVector(0, j1, 1), cz, cmplx.Conj(sz))
	Zrot(j1+1, b.CVector(0, j1-1, 1), b.CVector(0, j1, 1), cz, cmplx.Conj(sz))
	Zrot(n-j1+1, a.CVector(j1-1, j1-1), a.CVector(j1, j1-1), cq, sq)
	Zrot(n-j1+1, b.CVector(j1-1, j1-1), b.CVector(j1, j1-1), cq, sq)

	//     Set  N1 by N2 (2,1) blocks to 0
	a.Set(j1, j1-1, czero)
	b.Set(j1, j1-1, czero)

	//     Accumulate transformations into Q and Z if requested.
	if wantz {
		Zrot(n, z.CVector(0, j1-1, 1), z.CVector(0, j1, 1), cz, cmplx.Conj(sz))
	}
	if wantq {
		Zrot(n, q.CVector(0, j1-1, 1), q.CVector(0, j1, 1), cq, cmplx.Conj(sq))
	}

	//     Exit with INFO = 0 if swap was successfully performed.
	return

	//     Exit with INFO = 1 if swap was rejected.
label20:
	;
	info = 1

	return
}
