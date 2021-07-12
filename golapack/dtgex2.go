package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
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
func Dtgex2(wantq, wantz bool, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, q *mat.Matrix, ldq *int, z *mat.Matrix, ldz, j1, n1, n2 *int, work *mat.Vector, lwork, info *int) {
	var dtrong, wands, weak bool
	var bqra21, brqa21, ddum, dnorm, dscale, dsum, eps, f, g, one, sa, sb, scale, smlnum, ss, thresh, twenty, ws, zero float64
	var i, idum, ldst, linfo, m int
	var err error
	_ = err

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

	(*info) = 0

	//     Quick return if possible
	if (*n) <= 1 || (*n1) <= 0 || (*n2) <= 0 {
		return
	}
	if (*n1) > (*n) || ((*j1)+(*n1)) > (*n) {
		return
	}
	m = (*n1) + (*n2)
	if (*lwork) < max(1, (*n)*m, m*m*2) {
		(*info) = -16
		work.Set(0, float64(max(1, (*n)*m, m*m*2)))
		return
	}

	weak = false
	dtrong = false

	//     Make a local copy of selected block
	Dlaset('F', &ldst, &ldst, &zero, &zero, li, &ldst)
	Dlaset('F', &ldst, &ldst, &zero, &zero, ir, &ldst)
	Dlacpy('F', &m, &m, a.Off((*j1)-1, (*j1)-1), lda, s, &ldst)
	Dlacpy('F', &m, &m, b.Off((*j1)-1, (*j1)-1), ldb, t, &ldst)

	//     Compute threshold for testing acceptance of swapping.
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum) / eps
	dscale = zero
	dsum = one
	Dlacpy('F', &m, &m, s, &ldst, work.Matrix(m, opts), &m)
	Dlassq(toPtr(m*m), work, func() *int { y := 1; return &y }(), &dscale, &dsum)
	Dlacpy('F', &m, &m, t, &ldst, work.Matrix(m, opts), &m)
	Dlassq(toPtr(m*m), work, func() *int { y := 1; return &y }(), &dscale, &dsum)
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
		Dlartg(&f, &g, ir.GetPtr(0, 1), ir.GetPtr(0, 0), &ddum)
		ir.Set(1, 0, -ir.Get(0, 1))
		ir.Set(1, 1, ir.Get(0, 0))
		goblas.Drot(2, s.Vector(0, 0, 1), s.Vector(0, 1, 1), ir.Get(0, 0), ir.Get(1, 0))
		goblas.Drot(2, t.Vector(0, 0, 1), t.Vector(0, 1, 1), ir.Get(0, 0), ir.Get(1, 0))
		if sa >= sb {
			Dlartg(s.GetPtr(0, 0), s.GetPtr(1, 0), li.GetPtr(0, 0), li.GetPtr(1, 0), &ddum)
		} else {
			Dlartg(t.GetPtr(0, 0), t.GetPtr(1, 0), li.GetPtr(0, 0), li.GetPtr(1, 0), &ddum)
		}
		goblas.Drot(2, s.Vector(0, 0, ldst), s.Vector(1, 0, ldst), li.Get(0, 0), li.Get(1, 0))
		goblas.Drot(2, t.Vector(0, 0, ldst), t.Vector(1, 0, ldst), li.Get(0, 0), li.Get(1, 0))
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
			Dlacpy('F', &m, &m, a.Off((*j1)-1, (*j1)-1), lda, work.MatrixOff(m*m, m, opts), &m)
			err = goblas.Dgemm(NoTrans, NoTrans, m, m, m, one, li, s, zero, work.Matrix(m, opts))
			err = goblas.Dgemm(NoTrans, Trans, m, m, m, -one, work.Matrix(m, opts), ir, one, work.MatrixOff(m*m, m, opts))
			dscale = zero
			dsum = one
			Dlassq(toPtr(m*m), work.Off(m*m), func() *int { y := 1; return &y }(), &dscale, &dsum)

			Dlacpy('F', &m, &m, b.Off((*j1)-1, (*j1)-1), ldb, work.MatrixOff(m*m, m, opts), &m)
			err = goblas.Dgemm(NoTrans, NoTrans, m, m, m, one, li, t, zero, work.Matrix(m, opts))
			err = goblas.Dgemm(NoTrans, Trans, m, m, m, -one, work.Matrix(m, opts), ir, one, work.MatrixOff(m*m, m, opts))
			Dlassq(toPtr(m*m), work.Off(m*m), func() *int { y := 1; return &y }(), &dscale, &dsum)
			ss = dscale * math.Sqrt(dsum)
			dtrong = ss <= thresh
			if !dtrong {
				goto label70
			}
		}

		//        Update (A(J1:J1+M-1, M+J1:N), B(J1:J1+M-1, M+J1:N)) and
		//               (A(1:J1-1, J1:J1+M), B(1:J1-1, J1:J1+M)).
		goblas.Drot((*j1)+1, a.Vector(0, (*j1)-1, 1), a.Vector(0, (*j1), 1), ir.Get(0, 0), ir.Get(1, 0))
		goblas.Drot((*j1)+1, b.Vector(0, (*j1)-1, 1), b.Vector(0, (*j1), 1), ir.Get(0, 0), ir.Get(1, 0))
		goblas.Drot((*n)-(*j1)+1, a.Vector((*j1)-1, (*j1)-1, *lda), a.Vector((*j1), (*j1)-1, *lda), li.Get(0, 0), li.Get(1, 0))
		goblas.Drot((*n)-(*j1)+1, b.Vector((*j1)-1, (*j1)-1, *ldb), b.Vector((*j1), (*j1)-1, *ldb), li.Get(0, 0), li.Get(1, 0))

		//        Set  N1-by-N2 (2,1) - blocks to ZERO.
		a.Set((*j1), (*j1)-1, zero)
		b.Set((*j1), (*j1)-1, zero)

		//        Accumulate transformations into Q and Z if requested.
		if wantz {
			goblas.Drot(*n, z.Vector(0, (*j1)-1, 1), z.Vector(0, (*j1), 1), ir.Get(0, 0), ir.Get(1, 0))
		}
		if wantq {
			goblas.Drot(*n, q.Vector(0, (*j1)-1, 1), q.Vector(0, (*j1), 1), li.Get(0, 0), li.Get(1, 0))
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
		Dlacpy('F', n1, n2, t.Off(0, (*n1)), &ldst, li, &ldst)
		Dlacpy('F', n1, n2, s.Off(0, (*n1)), &ldst, ir.Off((*n2), (*n1)), &ldst)
		Dtgsy2('N', func() *int { y := 0; return &y }(), n1, n2, s, &ldst, s.Off((*n1), (*n1)), &ldst, ir.Off((*n2), (*n1)), &ldst, t, &ldst, t.Off((*n1), (*n1)), &ldst, li, &ldst, &scale, &dsum, &dscale, &iwork, &idum, &linfo)

		//        Compute orthogonal matrix QL:
		//
		//                    QL**T * LI = [ TL ]
		//                                 [ 0  ]
		//        where
		//                    LI =  [      -L              ]
		//                          [ SCALE * identity(N2) ]
		for i = 1; i <= (*n2); i++ {
			goblas.Dscal(*n1, -one, li.Vector(0, i-1, 1))
			li.Set((*n1)+i-1, i-1, scale)
		}
		Dgeqr2(&m, n2, li, &ldst, taul, work, &linfo)
		if linfo != 0 {
			goto label70
		}
		Dorg2r(&m, &m, n2, li, &ldst, taul, work, &linfo)
		if linfo != 0 {
			goto label70
		}

		//        Compute orthogonal matrix RQ:
		//
		//                    IR * RQ**T =   [ 0  TR],
		//
		//         where IR = [ SCALE * identity(N1), R ]
		for i = 1; i <= (*n1); i++ {
			ir.Set((*n2)+i-1, i-1, scale)
		}
		Dgerq2(n1, &m, ir.Off((*n2), 0), &ldst, taur, work, &linfo)
		if linfo != 0 {
			goto label70
		}
		Dorgr2(&m, &m, n1, ir, &ldst, taur, work, &linfo)
		if linfo != 0 {
			goto label70
		}

		//        Perform the swapping tentatively:
		err = goblas.Dgemm(Trans, NoTrans, m, m, m, one, li, s, zero, work.Matrix(m, opts))
		err = goblas.Dgemm(NoTrans, Trans, m, m, m, one, work.Matrix(m, opts), ir, zero, s)
		err = goblas.Dgemm(Trans, NoTrans, m, m, m, one, li, t, zero, work.Matrix(m, opts))
		err = goblas.Dgemm(NoTrans, Trans, m, m, m, one, work.Matrix(m, opts), ir, zero, t)
		Dlacpy('F', &m, &m, s, &ldst, scpy, &ldst)
		Dlacpy('F', &m, &m, t, &ldst, tcpy, &ldst)
		Dlacpy('F', &m, &m, ir, &ldst, ircop, &ldst)
		Dlacpy('F', &m, &m, li, &ldst, licop, &ldst)

		//        Triangularize the B-part by an RQ factorization.
		//        Apply transformation (from left) to A-part, giving S.
		Dgerq2(&m, &m, t, &ldst, taur, work, &linfo)
		if linfo != 0 {
			goto label70
		}
		Dormr2('R', 'T', &m, &m, &m, t, &ldst, taur, s, &ldst, work, &linfo)
		if linfo != 0 {
			goto label70
		}
		Dormr2('L', 'N', &m, &m, &m, t, &ldst, taur, ir, &ldst, work, &linfo)
		if linfo != 0 {
			goto label70
		}

		//        Compute F-norm(S21) in BRQA21. (T21 is 0.)
		dscale = zero
		dsum = one
		for i = 1; i <= (*n2); i++ {
			Dlassq(n1, s.Vector((*n2), i-1), func() *int { y := 1; return &y }(), &dscale, &dsum)
		}
		brqa21 = dscale * math.Sqrt(dsum)

		//        Triangularize the B-part by a QR factorization.
		//        Apply transformation (from right) to A-part, giving S.
		Dgeqr2(&m, &m, tcpy, &ldst, taul, work, &linfo)
		if linfo != 0 {
			goto label70
		}
		Dorm2r('L', 'T', &m, &m, &m, tcpy, &ldst, taul, scpy, &ldst, work, info)
		Dorm2r('R', 'N', &m, &m, &m, tcpy, &ldst, taul, licop, &ldst, work, info)
		if linfo != 0 {
			goto label70
		}

		//        Compute F-norm(S21) in BQRA21. (T21 is 0.)
		dscale = zero
		dsum = one
		for i = 1; i <= (*n2); i++ {
			Dlassq(n1, scpy.Vector((*n2), i-1), func() *int { y := 1; return &y }(), &dscale, &dsum)
		}
		bqra21 = dscale * math.Sqrt(dsum)

		//        Decide which method to use.
		//          Weak stability test:
		//             F-norm(S21) <= O(EPS * F-norm((S, T)))
		if bqra21 <= brqa21 && bqra21 <= thresh {
			Dlacpy('F', &m, &m, scpy, &ldst, s, &ldst)
			Dlacpy('F', &m, &m, tcpy, &ldst, t, &ldst)
			Dlacpy('F', &m, &m, ircop, &ldst, ir, &ldst)
			Dlacpy('F', &m, &m, licop, &ldst, li, &ldst)
		} else if brqa21 >= thresh {
			goto label70
		}

		//        Set lower triangle of B-part to zero
		Dlaset('L', toPtr(m-1), toPtr(m-1), &zero, &zero, t.Off(1, 0), &ldst)

		if wands {
			//           Strong stability test:
			//              F-norm((A-QL*S*QR**T, B-QL*T*QR**T)) <= O(EPS*F-norm((A,B)))
			Dlacpy('F', &m, &m, a.Off((*j1)-1, (*j1)-1), lda, work.MatrixOff(m*m, m, opts), &m)
			err = goblas.Dgemm(NoTrans, NoTrans, m, m, m, one, li, s, zero, work.Matrix(m, opts))
			err = goblas.Dgemm(NoTrans, NoTrans, m, m, m, -one, work.Matrix(m, opts), ir, one, work.MatrixOff(m*m, m, opts))
			dscale = zero
			dsum = one
			Dlassq(toPtr(m*m), work.Off(m*m), func() *int { y := 1; return &y }(), &dscale, &dsum)

			Dlacpy('F', &m, &m, b.Off((*j1)-1, (*j1)-1), ldb, work.MatrixOff(m*m, m, opts), &m)
			err = goblas.Dgemm(NoTrans, NoTrans, m, m, m, one, li, t, zero, work.Matrix(m, opts))
			err = goblas.Dgemm(NoTrans, NoTrans, m, m, m, -one, work.Matrix(m, opts), ir, one, work.MatrixOff(m*m, m, opts))
			Dlassq(toPtr(m*m), work.Off(m*m), func() *int { y := 1; return &y }(), &dscale, &dsum)
			ss = dscale * math.Sqrt(dsum)
			dtrong = (ss <= thresh)
			if !dtrong {
				goto label70
			}

		}

		//        If the swap is accepted ("weakly" and "strongly"), apply the
		//        transformations and set N1-by-N2 (2,1)-block to zero.
		Dlaset('F', n1, n2, &zero, &zero, s.Off((*n2), 0), &ldst)

		//        copy back M-by-M diagonal block starting at index J1 of (A, B)
		Dlacpy('F', &m, &m, s, &ldst, a.Off((*j1)-1, (*j1)-1), lda)
		Dlacpy('F', &m, &m, t, &ldst, b.Off((*j1)-1, (*j1)-1), ldb)
		Dlaset('F', &ldst, &ldst, &zero, &zero, t, &ldst)

		//        Standardize existing 2-by-2 blocks.
		Dlaset('F', &m, &m, &zero, &zero, work.Matrix(m, opts), &m)
		work.Set(0, one)
		t.Set(0, 0, one)
		idum = (*lwork) - m*m - 2
		if (*n2) > 1 {
			Dlagv2(a.Off((*j1)-1, (*j1)-1), lda, b.Off((*j1)-1, (*j1)-1), ldb, ar, ai, be, work.GetPtr(0), work.GetPtr(1), t.GetPtr(0, 0), t.GetPtr(1, 0))
			work.Set(m, -work.Get(1))
			work.Set(m+2-1, work.Get(0))
			t.Set((*n2)-1, (*n2)-1, t.Get(0, 0))
			t.Set(0, 1, -t.Get(1, 0))
		}
		work.Set(m*m-1, one)
		t.Set(m-1, m-1, one)
		//
		if (*n1) > 1 {
			Dlagv2(a.Off((*j1)+(*n2)-1, (*j1)+(*n2)-1), lda, b.Off((*j1)+(*n2)-1, (*j1)+(*n2)-1), ldb, taur, taul, work.Off(m*m), work.GetPtr((*n2)*m+(*n2)), work.GetPtr((*n2)*m+(*n2)+2-1), t.GetPtr((*n2), (*n2)), t.GetPtr(m-1, m-1-1))
			work.Set(m*m-1, work.Get((*n2)*m+(*n2)))
			work.Set(m*m-1-1, -work.Get((*n2)*m+(*n2)+2-1))
			t.Set(m-1, m-1, t.Get((*n2), (*n2)))
			t.Set(m-1-1, m-1, -t.Get(m-1, m-1-1))
		}
		err = goblas.Dgemm(Trans, NoTrans, *n2, *n1, *n2, one, work.Matrix(m, opts), a.Off((*j1)-1, (*j1)+(*n2)-1), zero, work.MatrixOff(m*m, *n2, opts))
		Dlacpy('F', n2, n1, work.MatrixOff(m*m, *n2, opts), n2, a.Off((*j1)-1, (*j1)+(*n2)-1), lda)
		err = goblas.Dgemm(Trans, NoTrans, *n2, *n1, *n2, one, work.Matrix(m, opts), b.Off((*j1)-1, (*j1)+(*n2)-1), zero, work.MatrixOff(m*m, *n2, opts))
		Dlacpy('F', n2, n1, work.MatrixOff(m*m, *n2, opts), n2, b.Off((*j1)-1, (*j1)+(*n2)-1), ldb)
		err = goblas.Dgemm(NoTrans, NoTrans, m, m, m, one, li, work.Matrix(m, opts), zero, work.MatrixOff(m*m, m, opts))
		Dlacpy('F', &m, &m, work.MatrixOff(m*m, m, opts), &m, li, &ldst)
		err = goblas.Dgemm(NoTrans, NoTrans, *n2, *n1, *n1, one, a.Off((*j1)-1, (*j1)+(*n2)-1), t.Off((*n2), (*n2)), zero, work.Matrix(*n2, opts))
		Dlacpy('F', n2, n1, work.Matrix(*n2, opts), n2, a.Off((*j1)-1, (*j1)+(*n2)-1), lda)
		err = goblas.Dgemm(NoTrans, NoTrans, *n2, *n1, *n1, one, b.Off((*j1)-1, (*j1)+(*n2)-1), t.Off((*n2), (*n2)), zero, work.Matrix(*n2, opts))
		Dlacpy('F', n2, n1, work.Matrix(*n2, opts), n2, b.Off((*j1)-1, (*j1)+(*n2)-1), ldb)
		err = goblas.Dgemm(Trans, NoTrans, m, m, m, one, ir, t, zero, work.Matrix(m, opts))
		Dlacpy('F', &m, &m, work.Matrix(m, opts), &m, ir, &ldst)

		//        Accumulate transformations into Q and Z if requested.
		if wantq {
			err = goblas.Dgemm(NoTrans, NoTrans, *n, m, m, one, q.Off(0, (*j1)-1), li, zero, work.Matrix(*n, opts))
			Dlacpy('F', n, &m, work.Matrix(*n, opts), n, q.Off(0, (*j1)-1), ldq)

		}

		if wantz {
			err = goblas.Dgemm(NoTrans, NoTrans, *n, m, m, one, z.Off(0, (*j1)-1), ir, zero, work.Matrix(*n, opts))
			Dlacpy('F', n, &m, work.Matrix(*n, opts), n, z.Off(0, (*j1)-1), ldz)

		}

		//        Update (A(J1:J1+M-1, M+J1:N), B(J1:J1+M-1, M+J1:N)) and
		//                (A(1:J1-1, J1:J1+M), B(1:J1-1, J1:J1+M)).
		i = (*j1) + m
		if i <= (*n) {
			err = goblas.Dgemm(Trans, NoTrans, m, (*n)-i+1, m, one, li, a.Off((*j1)-1, i-1), zero, work.Matrix(m, opts))
			Dlacpy('F', &m, toPtr((*n)-i+1), work.Matrix(m, opts), &m, a.Off((*j1)-1, i-1), lda)
			err = goblas.Dgemm(Trans, NoTrans, m, (*n)-i+1, m, one, li, b.Off((*j1)-1, i-1), zero, work.Matrix(m, opts))
			Dlacpy('F', &m, toPtr((*n)-i+1), work.Matrix(m, opts), &m, b.Off((*j1)-1, i-1), ldb)
		}
		i = (*j1) - 1
		if i > 0 {
			err = goblas.Dgemm(NoTrans, NoTrans, i, m, m, one, a.Off(0, (*j1)-1), ir, zero, work.Matrix(i, opts))
			Dlacpy('F', &i, &m, work.Matrix(i, opts), &i, a.Off(0, (*j1)-1), lda)
			err = goblas.Dgemm(NoTrans, NoTrans, i, m, m, one, b.Off(0, (*j1)-1), ir, zero, work.Matrix(i, opts))
			Dlacpy('F', &i, &m, work.Matrix(i, opts), &i, b.Off(0, (*j1)-1), ldb)
		}

		//        Exit with INFO = 0 if swap was successfully performed.
		return

	}

	//     Exit with INFO = 1 if swap was rejected.
label70:
	;

	(*info) = 1
}
