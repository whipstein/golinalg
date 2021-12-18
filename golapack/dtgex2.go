package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dtgex2 swaps adjacent diagonal blocks (A11, B11) and (A22, B22)
// of size 1-by-1 or 2-by-2 in an upper (quasi) triangular matrix pair
// (A, B) by an orthogonal equivalence transformation.
//
// (A, B) must be in generalized real Schur canonical form (as returned
// by DGGES), i.e. A is block upper triangular with 1-by-1 and 2-by-2
// diagonal blocks. B is upper triangular.
//
// Optionally, the matrices Q and Z of generalized Schur vectors are
// updated.
//
//        Q(in) * A(in) * Z(in)**T = Q(out) * A(out) * Z(out)**T
//        Q(in) * B(in) * Z(in)**T = Q(out) * B(out) * Z(out)**T
func Dtgex2(wantq, wantz bool, n int, a, b, q, z *mat.Matrix, j1, n1, n2 int, work *mat.Vector, lwork int) (info int, err error) {
	var dtrong, wands, weak bool
	var bqra21, brqa21, dnorm, dscale, dsum, eps, f, g, one, sa, sb, scale, smlnum, ss, thresh, twenty, ws, zero float64
	var i, ldst, linfo, m int

	ai := vf(2)
	ar := vf(2)
	be := vf(2)
	taul := vf(4)
	taur := vf(4)
	iwork := make([]int, 4)
	ir := mf(4, 4, opts)
	ircop := mf(4, 4, opts)
	li := mf(4, 4, opts)
	licop := mf(4, 4, opts)
	s := mf(4, 4, opts)
	scpy := mf(4, 4, opts)
	t := mf(4, 4, opts)
	tcpy := mf(4, 4, opts)

	zero = 0.0
	one = 1.0
	twenty = 2.0e+01
	ldst = 4
	wands = true

	//     Quick return if possible
	if n <= 1 || n1 <= 0 || n2 <= 0 {
		return
	}
	if n1 > n || (j1+n1) > n {
		return
	}
	m = n1 + n2
	if lwork < max(1, n*m, m*m*2) {
		err = fmt.Errorf("lwork < max(1, n*m, m*m*2): lwork=%v, m=%v, n=%v", lwork, m, n)
		info = -16
		work.Set(0, float64(max(1, n*m, m*m*2)))
		return
	}

	weak = false
	dtrong = false

	//     Make a local copy of selected block
	Dlaset(Full, ldst, ldst, zero, zero, li)
	Dlaset(Full, ldst, ldst, zero, zero, ir)
	Dlacpy(Full, m, m, a.Off(j1-1, j1-1), s)
	Dlacpy(Full, m, m, b.Off(j1-1, j1-1), t)

	//     Compute threshold for testing acceptance of swapping.
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum) / eps
	dscale = zero
	dsum = one
	Dlacpy(Full, m, m, s, work.Matrix(m, opts))
	dscale, dsum = Dlassq(m*m, work, 1, dscale, dsum)
	Dlacpy(Full, m, m, t, work.Matrix(m, opts))
	dscale, dsum = Dlassq(m*m, work, 1, dscale, dsum)
	dnorm = dscale * math.Sqrt(dsum)

	//     THRES has been changed from
	//        THRESH = MAX( TEN*EPS*SA, SMLNUM )
	//     to
	//        THRESH = MAX( TWENTY*EPS*SA, SMLNUM )
	//     on 04/01/10.
	//     "Bug" reported by Ondra Kamenik, confirmed by Julie Langou, fixed by
	//     Jim Demmel and Guillaume Revy. See forum post 1783.
	thresh = math.Max(twenty*eps*dnorm, smlnum)

	if m == 2 {
		//        CASE 1: Swap 1-by-1 and 1-by-1 blocks.
		//
		//        Compute orthogonal QL and RQ that swap 1-by-1 and 1-by-1 blocks
		//        using Givens rotations and perform the swap tentatively.
		f = s.Get(1, 1)*t.Get(0, 0) - t.Get(1, 1)*s.Get(0, 0)
		g = s.Get(1, 1)*t.Get(0, 1) - t.Get(1, 1)*s.Get(0, 1)
		sb = math.Abs(t.Get(1, 1))
		sa = math.Abs(s.Get(1, 1))
		*ir.GetPtr(0, 1), *ir.GetPtr(0, 0), _ = Dlartg(f, g)
		ir.Set(1, 0, -ir.Get(0, 1))
		ir.Set(1, 1, ir.Get(0, 0))
		s.Off(0, 1).Vector().Rot(2, s.Off(0, 0).Vector(), 1, 1, ir.Get(0, 0), ir.Get(1, 0))
		t.Off(0, 1).Vector().Rot(2, t.Off(0, 0).Vector(), 1, 1, ir.Get(0, 0), ir.Get(1, 0))
		if sa >= sb {
			*li.GetPtr(0, 0), *li.GetPtr(1, 0), _ = Dlartg(s.Get(0, 0), s.Get(1, 0))
		} else {
			*li.GetPtr(0, 0), *li.GetPtr(1, 0), _ = Dlartg(t.Get(0, 0), t.Get(1, 0))
		}
		s.Off(1, 0).Vector().Rot(2, s.Off(0, 0).Vector(), ldst, ldst, li.Get(0, 0), li.Get(1, 0))
		t.Off(1, 0).Vector().Rot(2, t.Off(0, 0).Vector(), ldst, ldst, li.Get(0, 0), li.Get(1, 0))
		li.Set(1, 1, li.Get(0, 0))
		li.Set(0, 1, -li.Get(1, 0))

		//        Weak stability test:
		//           |S21| + |T21| <= O(EPS * F-norm((S, T)))
		ws = math.Abs(s.Get(1, 0)) + math.Abs(t.Get(1, 0))
		weak = ws <= thresh
		if !weak {
			goto label70
		}

		if wands {
			//           Strong stability test:
			//             F-norm((A-QL**T*S*QR, B-QL**T*T*QR)) <= O(EPS*F-norm((A,B)))
			Dlacpy(Full, m, m, a.Off(j1-1, j1-1), work.Off(m*m).Matrix(m, opts))
			err = work.Matrix(m, opts).Gemm(NoTrans, NoTrans, m, m, m, one, li, s, zero)
			err = work.Off(m*m).Matrix(m, opts).Gemm(NoTrans, Trans, m, m, m, -one, work.Matrix(m, opts), ir, one)
			dscale = zero
			dsum = one
			dscale, dsum = Dlassq(m*m, work.Off(m*m), 1, dscale, dsum)

			Dlacpy(Full, m, m, b.Off(j1-1, j1-1), work.Off(m*m).Matrix(m, opts))
			err = work.Matrix(m, opts).Gemm(NoTrans, NoTrans, m, m, m, one, li, t, zero)
			err = work.Off(m*m).Matrix(m, opts).Gemm(NoTrans, Trans, m, m, m, -one, work.Matrix(m, opts), ir, one)
			dscale, dsum = Dlassq(m*m, work.Off(m*m), 1, dscale, dsum)
			ss = dscale * math.Sqrt(dsum)
			dtrong = ss <= thresh
			if !dtrong {
				goto label70
			}
		}

		//        Update (A(J1:J1+M-1, M+J1:N), B(J1:J1+M-1, M+J1:N)) and
		//               (A(1:J1-1, J1:J1+M), B(1:J1-1, J1:J1+M)).
		a.Off(0, j1).Vector().Rot(j1+1, a.Off(0, j1-1).Vector(), 1, 1, ir.Get(0, 0), ir.Get(1, 0))
		b.Off(0, j1).Vector().Rot(j1+1, b.Off(0, j1-1).Vector(), 1, 1, ir.Get(0, 0), ir.Get(1, 0))
		a.Off(j1, j1-1).Vector().Rot(n-j1+1, a.Off(j1-1, j1-1).Vector(), a.Rows, a.Rows, li.Get(0, 0), li.Get(1, 0))
		b.Off(j1, j1-1).Vector().Rot(n-j1+1, b.Off(j1-1, j1-1).Vector(), b.Rows, b.Rows, li.Get(0, 0), li.Get(1, 0))

		//        Set  N1-by-N2 (2,1) - blocks to ZERO.
		a.Set(j1, j1-1, zero)
		b.Set(j1, j1-1, zero)

		//        Accumulate transformations into Q and Z if requested.
		if wantz {
			z.Off(0, j1).Vector().Rot(n, z.Off(0, j1-1).Vector(), 1, 1, ir.Get(0, 0), ir.Get(1, 0))
		}
		if wantq {
			q.Off(0, j1).Vector().Rot(n, q.Off(0, j1-1).Vector(), 1, 1, li.Get(0, 0), li.Get(1, 0))
		}

		//        Exit with INFO = 0 if swap was successfully performed.
		return

	} else {
		//        CASE 2: Swap 1-by-1 and 2-by-2 blocks, or 2-by-2
		//                and 2-by-2 blocks.
		//
		//        Solve the generalized Sylvester equation
		//                 S11 * R - L * S22 = SCALE * S12
		//                 T11 * R - L * T22 = SCALE * T12
		//        for R and L. Solutions in LI and IR.
		Dlacpy(Full, n1, n2, t.Off(0, n1), li)
		Dlacpy(Full, n1, n2, s.Off(0, n1), ir.Off(n2, n1))
		if scale, dsum, dscale, _, _, err = Dtgsy2(NoTrans, 0, n1, n2, s, s.Off(n1, n1), ir.Off(n2, n1), t, t.Off(n1, n1), li, dsum, dscale, &iwork); err != nil {
			panic(err)
		}

		//        Compute orthogonal matrix QL:
		//
		//                    QL**T * LI = [ TL ]
		//                                 [ 0  ]
		//        where
		//                    LI =  [      -L              ]
		//                          [ SCALE * identity(N2) ]
		for i = 1; i <= n2; i++ {
			li.Off(0, i-1).Vector().Scal(n1, -one, 1)
			li.Set(n1+i-1, i-1, scale)
		}
		if err = Dgeqr2(m, n2, li, taul, work); err != nil {
			goto label70
		}
		if err = Dorg2r(m, m, n2, li, taul, work); err != nil {
			goto label70
		}

		//        Compute orthogonal matrix RQ:
		//
		//                    IR * RQ**T =   [ 0  TR],
		//
		//         where IR = [ SCALE * identity(N1), R ]
		for i = 1; i <= n1; i++ {
			ir.Set(n2+i-1, i-1, scale)
		}
		if err = Dgerq2(n1, m, ir.Off(n2, 0), taur, work); err != nil {
			panic(err)
		}
		if linfo != 0 {
			goto label70
		}
		if err = Dorgr2(m, m, n1, ir, taur, work); err != nil {
			panic(err)
		}
		if linfo != 0 {
			goto label70
		}

		//        Perform the swapping tentatively:
		if err = work.Matrix(m, opts).Gemm(Trans, NoTrans, m, m, m, one, li, s, zero); err != nil {
			panic(err)
		}
		if err = s.Gemm(NoTrans, Trans, m, m, m, one, work.Matrix(m, opts), ir, zero); err != nil {
			panic(err)
		}
		if err = work.Matrix(m, opts).Gemm(Trans, NoTrans, m, m, m, one, li, t, zero); err != nil {
			panic(err)
		}
		if err = t.Gemm(NoTrans, Trans, m, m, m, one, work.Matrix(m, opts), ir, zero); err != nil {
			panic(err)
		}
		Dlacpy(Full, m, m, s, scpy)
		Dlacpy(Full, m, m, t, tcpy)
		Dlacpy(Full, m, m, ir, ircop)
		Dlacpy(Full, m, m, li, licop)

		//        Triangularize the B-part by an RQ factorization.
		//        Apply transformation (from left) to A-part, giving S.
		if err = Dgerq2(m, m, t, taur, work); err != nil {
			goto label70
		}
		if err = Dormr2(Right, Trans, m, m, m, t, taur, s, work); err != nil {
			goto label70
		}
		if err = Dormr2(Left, NoTrans, m, m, m, t, taur, ir, work); err != nil {
			goto label70
		}

		//        Compute F-norm(S21) in BRQA21. (T21 is 0.)
		dscale = zero
		dsum = one
		for i = 1; i <= n2; i++ {
			dscale, dsum = Dlassq(n1, s.Off(n2, i-1).Vector(), 1, dscale, dsum)
		}
		brqa21 = dscale * math.Sqrt(dsum)

		//        Triangularize the B-part by a QR factorization.
		//        Apply transformation (from right) to A-part, giving S.
		if err = Dgeqr2(m, m, tcpy, taul, work); err != nil {
			goto label70
		}
		if err = Dorm2r(Left, Trans, m, m, m, tcpy, taul, scpy, work); err != nil {
			panic(err)
		}
		if err = Dorm2r(Right, NoTrans, m, m, m, tcpy, taul, licop, work); err != nil {
			panic(err)
		}
		if linfo != 0 {
			goto label70
		}

		//        Compute F-norm(S21) in BQRA21. (T21 is 0.)
		dscale = zero
		dsum = one
		for i = 1; i <= n2; i++ {
			dscale, dsum = Dlassq(n1, scpy.Off(n2, i-1).Vector(), 1, dscale, dsum)
		}
		bqra21 = dscale * math.Sqrt(dsum)

		//        Decide which method to use.
		//          Weak stability test:
		//             F-norm(S21) <= O(EPS * F-norm((S, T)))
		if bqra21 <= brqa21 && bqra21 <= thresh {
			Dlacpy(Full, m, m, scpy, s)
			Dlacpy(Full, m, m, tcpy, t)
			Dlacpy(Full, m, m, ircop, ir)
			Dlacpy(Full, m, m, licop, li)
		} else if brqa21 >= thresh {
			goto label70
		}

		//        Set lower triangle of B-part to zero
		Dlaset(Lower, m-1, m-1, zero, zero, t.Off(1, 0))

		if wands {
			//           Strong stability test:
			//              F-norm((A-QL*S*QR**T, B-QL*T*QR**T)) <= O(EPS*F-norm((A,B)))
			Dlacpy(Full, m, m, a.Off(j1-1, j1-1), work.Off(m*m).Matrix(m, opts))
			err = work.Matrix(m, opts).Gemm(NoTrans, NoTrans, m, m, m, one, li, s, zero)
			err = work.Off(m*m).Matrix(m, opts).Gemm(NoTrans, NoTrans, m, m, m, -one, work.Matrix(m, opts), ir, one)
			dscale = zero
			dsum = one
			dscale, dsum = Dlassq(m*m, work.Off(m*m), 1, dscale, dsum)

			Dlacpy(Full, m, m, b.Off(j1-1, j1-1), work.Off(m*m).Matrix(m, opts))
			err = work.Matrix(m, opts).Gemm(NoTrans, NoTrans, m, m, m, one, li, t, zero)
			err = work.Off(m*m).Matrix(m, opts).Gemm(NoTrans, NoTrans, m, m, m, -one, work.Matrix(m, opts), ir, one)
			dscale, dsum = Dlassq(m*m, work.Off(m*m), 1, dscale, dsum)
			ss = dscale * math.Sqrt(dsum)
			dtrong = (ss <= thresh)
			if !dtrong {
				goto label70
			}

		}

		//        If the swap is accepted ("weakly" and "strongly"), apply the
		//        transformations and set N1-by-N2 (2,1)-block to zero.
		Dlaset(Full, n1, n2, zero, zero, s.Off(n2, 0))

		//        copy back M-by-M diagonal block starting at index J1 of (A, B)
		Dlacpy(Full, m, m, s, a.Off(j1-1, j1-1))
		Dlacpy(Full, m, m, t, b.Off(j1-1, j1-1))
		Dlaset(Full, ldst, ldst, zero, zero, t)

		//        Standardize existing 2-by-2 blocks.
		Dlaset(Full, m, m, zero, zero, work.Matrix(m, opts))
		work.Set(0, one)
		t.Set(0, 0, one)
		// idum = lwork - m*m - 2
		if n2 > 1 {
			*work.GetPtr(0), *work.GetPtr(1), *t.GetPtr(0, 0), *t.GetPtr(1, 0) = Dlagv2(a.Off(j1-1, j1-1), b.Off(j1-1, j1-1), ar, ai, be)
			work.Set(m, -work.Get(1))
			work.Set(m+2-1, work.Get(0))
			t.Set(n2-1, n2-1, t.Get(0, 0))
			t.Set(0, 1, -t.Get(1, 0))
		}
		work.Set(m*m-1, one)
		t.Set(m-1, m-1, one)
		//
		if n1 > 1 {
			*work.GetPtr(n2*m + n2), *work.GetPtr(n2*m + n2 + 2 - 1), *t.GetPtr(n2, n2), *t.GetPtr(m-1, m-1-1) = Dlagv2(a.Off(j1+n2-1, j1+n2-1), b.Off(j1+n2-1, j1+n2-1), taur, taul, work.Off(m*m))
			work.Set(m*m-1, work.Get(n2*m+n2))
			work.Set(m*m-1-1, -work.Get(n2*m+n2+2-1))
			t.Set(m-1, m-1, t.Get(n2, n2))
			t.Set(m-1-1, m-1, -t.Get(m-1, m-1-1))
		}
		err = work.Off(m*m).Matrix(n2, opts).Gemm(Trans, NoTrans, n2, n1, n2, one, work.Matrix(m, opts), a.Off(j1-1, j1+n2-1), zero)
		Dlacpy(Full, n2, n1, work.Off(m*m).Matrix(n2, opts), a.Off(j1-1, j1+n2-1))
		err = work.Off(m*m).Matrix(n2, opts).Gemm(Trans, NoTrans, n2, n1, n2, one, work.Matrix(m, opts), b.Off(j1-1, j1+n2-1), zero)
		Dlacpy(Full, n2, n1, work.Off(m*m).Matrix(n2, opts), b.Off(j1-1, j1+n2-1))
		err = work.Off(m*m).Matrix(m, opts).Gemm(NoTrans, NoTrans, m, m, m, one, li, work.Matrix(m, opts), zero)
		Dlacpy(Full, m, m, work.Off(m*m).Matrix(m, opts), li)
		err = work.Matrix(n2, opts).Gemm(NoTrans, NoTrans, n2, n1, n1, one, a.Off(j1-1, j1+n2-1), t.Off(n2, n2), zero)
		Dlacpy(Full, n2, n1, work.Matrix(n2, opts), a.Off(j1-1, j1+n2-1))
		err = work.Matrix(n2, opts).Gemm(NoTrans, NoTrans, n2, n1, n1, one, b.Off(j1-1, j1+n2-1), t.Off(n2, n2), zero)
		Dlacpy(Full, n2, n1, work.Matrix(n2, opts), b.Off(j1-1, j1+n2-1))
		err = work.Matrix(m, opts).Gemm(Trans, NoTrans, m, m, m, one, ir, t, zero)
		Dlacpy(Full, m, m, work.Matrix(m, opts), ir)

		//        Accumulate transformations into Q and Z if requested.
		if wantq {
			err = work.Matrix(n, opts).Gemm(NoTrans, NoTrans, n, m, m, one, q.Off(0, j1-1), li, zero)
			Dlacpy(Full, n, m, work.Matrix(n, opts), q.Off(0, j1-1))

		}

		if wantz {
			err = work.Matrix(n, opts).Gemm(NoTrans, NoTrans, n, m, m, one, z.Off(0, j1-1), ir, zero)
			Dlacpy(Full, n, m, work.Matrix(n, opts), z.Off(0, j1-1))

		}

		//        Update (A(J1:J1+M-1, M+J1:N), B(J1:J1+M-1, M+J1:N)) and
		//                (A(1:J1-1, J1:J1+M), B(1:J1-1, J1:J1+M)).
		i = j1 + m
		if i <= n {
			err = work.Matrix(m, opts).Gemm(Trans, NoTrans, m, n-i+1, m, one, li, a.Off(j1-1, i-1), zero)
			Dlacpy(Full, m, n-i+1, work.Matrix(m, opts), a.Off(j1-1, i-1))
			err = work.Matrix(m, opts).Gemm(Trans, NoTrans, m, n-i+1, m, one, li, b.Off(j1-1, i-1), zero)
			Dlacpy(Full, m, n-i+1, work.Matrix(m, opts), b.Off(j1-1, i-1))
		}
		i = j1 - 1
		if i > 0 {
			err = work.Matrix(i, opts).Gemm(NoTrans, NoTrans, i, m, m, one, a.Off(0, j1-1), ir, zero)
			Dlacpy(Full, i, m, work.Matrix(i, opts), a.Off(0, j1-1))
			err = work.Matrix(i, opts).Gemm(NoTrans, NoTrans, i, m, m, one, b.Off(0, j1-1), ir, zero)
			Dlacpy(Full, i, m, work.Matrix(i, opts), b.Off(0, j1-1))
		}

		//        Exit with INFO = 0 if swap was successfully performed.
		return

	}

	//     Exit with INFO = 1 if swap was rejected.
label70:
	;

	info = 1

	return
}
