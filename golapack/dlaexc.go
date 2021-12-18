package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlaexc swaps adjacent diagonal blocks T11 and T22 of order 1 or 2 in
// an upper quasi-triangular matrix T by an orthogonal similarity
// transformation.
//
// T must be in Schur canonical form, that is, block upper triangular
// with 1-by-1 and 2-by-2 diagonal blocks; each 2-by-2 diagonal block
// has its diagonal elemnts equal and its off-diagonal elements of
// opposite sign.
func Dlaexc(wantq bool, n int, t, q *mat.Matrix, j1, n1, n2 int, work *mat.Vector) (info int) {
	var cs, dnorm, eps, one, scale, smlnum, sn, t11, t22, t33, tau, tau1, tau2, temp, ten, thresh, zero float64
	var j2, j3, j4, k, nd int

	u := vf(3)
	u1 := vf(3)
	u2 := vf(3)
	d := mf(4, 4, opts)
	x := mf(2, 2, opts)

	zero = 0.0
	one = 1.0
	ten = 1.0e+1

	//     Quick return if possible
	if n == 0 || n1 == 0 || n2 == 0 {
		return
	}
	if j1+n1 > n {
		return
	}

	j2 = j1 + 1
	j3 = j1 + 2
	j4 = j1 + 3

	if n1 == 1 && n2 == 1 {
		//        Swap two 1-by-1 blocks.
		t11 = t.Get(j1-1, j1-1)
		t22 = t.Get(j2-1, j2-1)

		//        Determine the transformation to perform the interchange.
		cs, sn, temp = Dlartg(t.Get(j1-1, j2-1), t22-t11)

		//        Apply transformation to the matrix T.
		if j3 <= n {
			t.Off(j2-1, j3-1).Vector().Rot(n-j1-1, t.Off(j1-1, j3-1).Vector(), t.Rows, t.Rows, cs, sn)
		}
		t.Off(0, j2-1).Vector().Rot(j1-1, t.Off(0, j1-1).Vector(), 1, 1, cs, sn)

		t.Set(j1-1, j1-1, t22)
		t.Set(j2-1, j2-1, t11)

		if wantq {
			//           Accumulate transformation in the matrix Q.
			q.Off(0, j2-1).Vector().Rot(n, q.Off(0, j1-1).Vector(), 1, 1, cs, sn)
		}

	} else {
		//        Swapping involves at least one 2-by-2 block.
		//
		//        Copy the diagonal block of order N1+N2 to the local array D
		//        and compute its norm.
		nd = n1 + n2
		Dlacpy(Full, nd, nd, t.Off(j1-1, j1-1), d)
		dnorm = Dlange('M', nd, nd, d, work)

		//        Compute machine-dependent threshold for test for accepting
		//        swap.
		eps = Dlamch(Precision)
		smlnum = Dlamch(SafeMinimum) / eps
		thresh = math.Max(ten*eps*dnorm, smlnum)

		//        Solve T11*X - X*T22 = scale*T12 for X.
		scale, _, _ = Dlasy2(false, false, -1, n1, n2, d, d.Off(n1, n1), d.Off(0, n1), x)

		//        Swap the adjacent diagonal blocks.
		k = n1 + n1 + n2 - 3
		switch k {
		case 1:
			goto label10
		case 2:
			goto label20
		case 3:
			goto label30
		}

	label10:
		;

		//        N1 = 1, N2 = 2: generate elementary reflector H so that:
		//
		//        ( scale, X11, X12 ) H = ( 0, 0, * )
		u.Set(0, scale)
		u.Set(1, x.Get(0, 0))
		u.Set(2, x.Get(0, 1))
		*u.GetPtr(2), tau = Dlarfg(3, u.Get(2), u, 1)
		u.Set(2, one)
		t11 = t.Get(j1-1, j1-1)

		//        Perform swap provisionally on diagonal block in D.
		Dlarfx(Left, 3, 3, u, tau, d, work)
		Dlarfx(Right, 3, 3, u, tau, d, work)

		//        Test whether to reject swap.
		if math.Max(math.Abs(d.Get(2, 0)), math.Max(math.Abs(d.Get(2, 1)), math.Abs(d.Get(2, 2)-t11))) > thresh {
			goto label50
		}

		//        Accept swap: apply transformation to the entire matrix T.
		Dlarfx(Left, 3, n-j1+1, u, tau, t.Off(j1-1, j1-1), work)
		Dlarfx(Right, j2, 3, u, tau, t.Off(0, j1-1), work)

		t.Set(j3-1, j1-1, zero)
		t.Set(j3-1, j2-1, zero)
		t.Set(j3-1, j3-1, t11)

		if wantq {
			//           Accumulate transformation in the matrix Q.
			Dlarfx(Right, n, 3, u, tau, q.Off(0, j1-1), work)
		}
		goto label40

	label20:
		;

		//        N1 = 2, N2 = 1: generate elementary reflector H so that:
		//
		//        H (  -X11 ) = ( * )
		//          (  -X21 ) = ( 0 )
		//          ( scale ) = ( 0 )
		u.Set(0, -x.Get(0, 0))
		u.Set(1, -x.Get(1, 0))
		u.Set(2, scale)
		*u.GetPtr(0), tau = Dlarfg(3, u.Get(0), u.Off(1), 1)
		u.Set(0, one)
		t33 = t.Get(j3-1, j3-1)

		//        Perform swap provisionally on diagonal block in D.
		Dlarfx(Left, 3, 3, u, tau, d, work)
		Dlarfx(Right, 3, 3, u, tau, d, work)

		//        Test whether to reject swap.
		if math.Max(math.Abs(d.Get(1, 0)), math.Max(math.Abs(d.Get(2, 0)), math.Abs(d.Get(0, 0)-t33))) > thresh {
			goto label50
		}

		//        Accept swap: apply transformation to the entire matrix T.
		Dlarfx(Right, j3, 3, u, tau, t.Off(0, j1-1), work)
		Dlarfx(Left, 3, n-j1, u, tau, t.Off(j1-1, j2-1), work)

		t.Set(j1-1, j1-1, t33)
		t.Set(j2-1, j1-1, zero)
		t.Set(j3-1, j1-1, zero)

		if wantq {
			//           Accumulate transformation in the matrix Q.
			Dlarfx(Right, n, 3, u, tau, q.Off(0, j1-1), work)
		}
		goto label40

	label30:
		;

		//        N1 = 2, N2 = 2: generate elementary reflectors H(1) and H(2) so
		//        that:
		//
		//        H(2) H(1) (  -X11  -X12 ) = (  *  * )
		//                  (  -X21  -X22 )   (  0  * )
		//                  ( scale    0  )   (  0  0 )
		//                  (    0  scale )   (  0  0 )
		u1.Set(0, -x.Get(0, 0))
		u1.Set(1, -x.Get(1, 0))
		u1.Set(2, scale)
		*u1.GetPtr(0), tau1 = Dlarfg(3, u1.Get(0), u1.Off(1), 1)
		u1.Set(0, one)

		temp = -tau1 * (x.Get(0, 1) + u1.Get(1)*x.Get(1, 1))
		u2.Set(0, -temp*u1.Get(1)-x.Get(1, 1))
		u2.Set(1, -temp*u1.Get(2))
		u2.Set(2, scale)
		*u2.GetPtr(0), tau2 = Dlarfg(3, u2.Get(0), u2.Off(1), 1)
		u2.Set(0, one)

		//        Perform swap provisionally on diagonal block in D.
		Dlarfx(Left, 3, 4, u1, tau1, d, work)
		Dlarfx(Right, 4, 3, u1, tau1, d, work)
		Dlarfx(Left, 3, 4, u2, tau2, d.Off(1, 0), work)
		Dlarfx(Right, 4, 3, u2, tau2, d.Off(0, 1), work)

		//        Test whether to reject swap.
		if math.Max(math.Abs(d.Get(2, 0)), math.Max(math.Abs(d.Get(2, 1)), math.Max(math.Abs(d.Get(3, 0)), math.Abs(d.Get(3, 1))))) > thresh {
			goto label50
		}

		//        Accept swap: apply transformation to the entire matrix T.
		Dlarfx(Left, 3, n-j1+1, u1, tau1, t.Off(j1-1, j1-1), work)
		Dlarfx(Right, j4, 3, u1, tau1, t.Off(0, j1-1), work)
		Dlarfx(Left, 3, n-j1+1, u2, tau2, t.Off(j2-1, j1-1), work)
		Dlarfx(Right, j4, 3, u2, tau2, t.Off(0, j2-1), work)

		t.Set(j3-1, j1-1, zero)
		t.Set(j3-1, j2-1, zero)
		t.Set(j4-1, j1-1, zero)
		t.Set(j4-1, j2-1, zero)

		if wantq {
			//           Accumulate transformation in the matrix Q.
			Dlarfx(Right, n, 3, u1, tau1, q.Off(0, j1-1), work)
			Dlarfx(Right, n, 3, u2, tau2, q.Off(0, j2-1), work)
		}

	label40:
		;

		if n2 == 2 {
			//           Standardize new 2-by-2 block T11
			*t.GetPtr(j1-1, j1-1), *t.GetPtr(j1-1, j2-1), *t.GetPtr(j2-1, j1-1), *t.GetPtr(j2-1, j2-1), _, _, _, _, cs, sn = Dlanv2(t.Get(j1-1, j1-1), t.Get(j1-1, j2-1), t.Get(j2-1, j1-1), t.Get(j2-1, j2-1))
			t.Off(j2-1, j1+2-1).Vector().Rot(n-j1-1, t.Off(j1-1, j1+2-1).Vector(), t.Rows, t.Rows, cs, sn)
			t.Off(0, j2-1).Vector().Rot(j1-1, t.Off(0, j1-1).Vector(), 1, 1, cs, sn)
			if wantq {
				q.Off(0, j2-1).Vector().Rot(n, q.Off(0, j1-1).Vector(), 1, 1, cs, sn)
			}
		}

		if n1 == 2 {
			//           Standardize new 2-by-2 block T22
			j3 = j1 + n2
			j4 = j3 + 1
			*t.GetPtr(j3-1, j3-1), *t.GetPtr(j3-1, j4-1), *t.GetPtr(j4-1, j3-1), *t.GetPtr(j4-1, j4-1), _, _, _, _, cs, sn = Dlanv2(t.Get(j3-1, j3-1), t.Get(j3-1, j4-1), t.Get(j4-1, j3-1), t.Get(j4-1, j4-1))
			if j3+2 <= n {
				t.Off(j4-1, j3+2-1).Vector().Rot(n-j3-1, t.Off(j3-1, j3+2-1).Vector(), t.Rows, t.Rows, cs, sn)
			}
			t.Off(0, j4-1).Vector().Rot(j3-1, t.Off(0, j3-1).Vector(), 1, 1, cs, sn)
			if wantq {
				q.Off(0, j4-1).Vector().Rot(n, q.Off(0, j3-1).Vector(), 1, 1, cs, sn)
			}
		}

	}
	return

	//     Exit with INFO = 1 if swap was rejected.
label50:
	;
	info = 1

	return
}
