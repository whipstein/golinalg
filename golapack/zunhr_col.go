package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// ZunhrCol takes an M-by-N complex matrix Q_in with orthonormal columns
//  as input, stored in A, and performs Householder Reconstruction (HR),
//  i.e. reconstructs Householder vectors V(i) implicitly representing
//  another M-by-N matrix Q_out, with the property that Q_in = Q_out*S,
//  where S is an N-by-N diagonal matrix with diagonal entries
//  equal to +1 or -1. The Householder vectors (columns V(i) of V) are
//  stored in A on output, and the diagonal entries of S are stored in D.
//  Block reflectors are also returned in T
//  (same output format as ZGEQRT).
func ZunhrCol(m, n, nb int, a, t *mat.CMatrix, d *mat.CVector) (err error) {
	var cone, czero complex128
	var i, j, jb, jbtemp1, jbtemp2, jnb, nplusone int

	cone = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)

	//     Test the input parameters
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 || n > m {
		err = fmt.Errorf("n < 0 || n > m: m=%v, n=%v", m, n)
	} else if nb < 1 {
		err = fmt.Errorf("nb < 1: nb=%v", nb)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if t.Rows < max(1, min(nb, n)) {
		err = fmt.Errorf("t.Rows < max(1, min(nb, n)): t.Rows=%v, n=%v, nb=%v", t.Rows, n, nb)
	}

	//     Handle error in the input parameters.
	if err != nil {
		gltest.Xerbla2("ZunhrCol", err)
		return
	}

	//     Quick return if possible
	if min(m, n) == 0 {
		return
	}

	//     On input, the M-by-N matrix A contains the unitary
	//     M-by-N matrix Q_in.
	//
	//     (1) Compute the unit lower-trapezoidal V (ones on the diagonal
	//     are not stored) by performing the "modified" LU-decomposition.
	//
	//     Q_in - ( S ) = V * U = ( V1 ) * U,
	//            ( 0 )           ( V2 )
	//
	//     where 0 is an (M-N)-by-N zero matrix.
	//
	//     (1-1) Factor V1 and U.
	if err = ZlaunhrColGetrfnp(n, n, a, d); err != nil {
		panic(err)
	}

	//     (1-2) Solve for V2.
	if m > n {
		if err = a.Off(n, 0).Trsm(Right, Upper, NoTrans, NonUnit, m-n, n, cone, a); err != nil {
			panic(err)
		}
	}

	//     (2) Reconstruct the block reflector T stored in T(1:NB, 1:N)
	//     as a sequence of upper-triangular blocks with NB-size column
	//     blocking.
	//
	//     Loop over the column blocks of size NB of the array A(1:M,1:N)
	//     and the array T(1:NB,1:N), JB is the column index of a column
	//     block, JNB is the column block size at each step JB.
	nplusone = n + 1
	for jb = 1; jb <= n; jb += nb {
		//        (2-0) Determine the column block size JNB.
		jnb = min(nplusone-jb, nb)

		//        (2-1) Copy the upper-triangular part of the current JNB-by-JNB
		//        diagonal block U(JB) (of the N-by-N matrix U) stored
		//        in A(JB:JB+JNB-1,JB:JB+JNB-1) into the upper-triangular part
		//        of the current JNB-by-JNB block T(1:JNB,JB:JB+JNB-1)
		//        column-by-column, total JNB*(JNB+1)/2 elements.
		jbtemp1 = jb - 1
		for j = jb; j <= jb+jnb-1; j++ {
			t.Off(0, j-1).CVector().Copy(j-jbtemp1, a.Off(jb-1, j-1).CVector(), 1, 1)
		}

		//        (2-2) Perform on the upper-triangular part of the current
		//        JNB-by-JNB diagonal block U(JB) (of the N-by-N matrix U) stored
		//        in T(1:JNB,JB:JB+JNB-1) the following operation in place:
		//        (-1)*U(JB)*S(JB), i.e the result will be stored in the upper-
		//        triangular part of T(1:JNB,JB:JB+JNB-1). This multiplication
		//        of the JNB-by-JNB diagonal block U(JB) by the JNB-by-JNB
		//        diagonal block S(JB) of the N-by-N sign matrix S from the
		//        right means changing the sign of each J-th column of the block
		//        U(JB) according to the sign of the diagonal element of the block
		//        S(JB), i.e. S(J,J) that is stored in the array element D(J).
		for j = jb; j <= jb+jnb-1; j++ {
			if d.Get(j-1) == cone {
				t.Off(0, j-1).CVector().Scal(j-jbtemp1, -cone, 1)
			}
		}

		//        (2-3) Perform the triangular solve for the current block
		//        matrix X(JB):
		//
		//               X(JB) * (A(JB)**T) = B(JB), where:
		//
		//               A(JB)**T  is a JNB-by-JNB unit upper-triangular
		//                         coefficient block, and A(JB)=V1(JB), which
		//                         is a JNB-by-JNB unit lower-triangular block
		//                         stored in A(JB:JB+JNB-1,JB:JB+JNB-1).
		//                         The N-by-N matrix V1 is the upper part
		//                         of the M-by-N lower-trapezoidal matrix V
		//                         stored in A(1:M,1:N);
		//
		//               B(JB)     is a JNB-by-JNB  upper-triangular right-hand
		//                         side block, B(JB) = (-1)*U(JB)*S(JB), and
		//                         B(JB) is stored in T(1:JNB,JB:JB+JNB-1);
		//
		//               X(JB)     is a JNB-by-JNB upper-triangular solution
		//                         block, X(JB) is the upper-triangular block
		//                         reflector T(JB), and X(JB) is stored
		//                         in T(1:JNB,JB:JB+JNB-1).
		//
		//             In other words, we perform the triangular solve for the
		//             upper-triangular block T(JB):
		//
		//               T(JB) * (V1(JB)**T) = (-1)*U(JB)*S(JB).
		//
		//             Even though the blocks X(JB) and B(JB) are upper-
		//             triangular, the routine ZTRSM will access all JNB**2
		//             elements of the square T(1:JNB,JB:JB+JNB-1). Therefore,
		//             we need to set to zero the elements of the block
		//             T(1:JNB,JB:JB+JNB-1) below the diagonal before the call
		//             to ZTRSM.
		//
		//        (2-3a) Set the elements to zero.
		jbtemp2 = jb - 2
		for j = jb; j <= jb+jnb-2; j++ {
			for i = j - jbtemp2; i <= nb; i++ {
				t.Set(i-1, j-1, czero)
			}
		}

		//        (2-3b) Perform the triangular solve.
		if err = t.Off(0, jb-1).Trsm(Right, Lower, ConjTrans, Unit, jnb, jnb, cone, a.Off(jb-1, jb-1)); err != nil {
			panic(err)
		}

	}

	return
}
