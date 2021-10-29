package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgelqt3 recursively computes a LQ factorization of a complex M-by-N
// matrix A, using the compact WY representation of Q.
//
// Based on the algorithm of Elmroth and Gustavson,
// IBM J. Res. Develop. Vol 44 No. 4 July 2000.
func Zgelqt3(m, n int, a, t *mat.CMatrix) (err error) {
	var one, zero complex128
	var i, i1, j, j1, m1, m2 int

	one = (1.0e+00 + 0.0e+00*1i)
	zero = (0.0e+00 + 0.0e+00*1i)

	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < m {
		err = fmt.Errorf("n < m: m=%v, n=%v", m, n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if t.Rows < max(1, m) {
		err = fmt.Errorf("t.Rows < max(1, m): t.Rows=%v, m=%v", t.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("Zgelqt3", err)
		return
	}

	if m == 1 {
		//        Compute Householder transform when N=1
		*a.GetPtr(0, 0), *t.GetPtr(0, 0) = Zlarfg(n, a.Get(0, 0), a.CVector(0, min(2, n)-1))
		t.Set(0, 0, t.GetConj(0, 0))

	} else {
		//        Otherwise, split A into blocks...
		m1 = m / 2
		m2 = m - m1
		i1 = min(m1+1, m)
		j1 = min(m+1, n)

		//        Compute A(1:M1,1:N) <- (Y1,R1,T1), where Q1 = I - Y1 T1 Y1^H
		if err = Zgelqt3(m1, n, a, t); err != nil {
			panic(err)
		}

		//        Compute A(J1:M,1:N) =  A(J1:M,1:N) Q1^H [workspace: T(1:N1,J1:N)]
		for i = 1; i <= m2; i++ {
			for j = 1; j <= m1; j++ {
				t.Set(i+m1-1, j-1, a.Get(i+m1-1, j-1))
			}
		}
		if err = goblas.Ztrmm(Right, Upper, ConjTrans, Unit, m2, m1, one, a, t.Off(i1-1, 0)); err != nil {
			panic(err)
		}

		if err = goblas.Zgemm(NoTrans, ConjTrans, m2, m1, n-m1, one, a.Off(i1-1, i1-1), a.Off(0, i1-1), one, t.Off(i1-1, 0)); err != nil {
			panic(err)
		}

		if err = goblas.Ztrmm(Right, Upper, NoTrans, NonUnit, m2, m1, one, t, t.Off(i1-1, 0)); err != nil {
			panic(err)
		}

		if err = goblas.Zgemm(NoTrans, NoTrans, m2, n-m1, m1, -one, t.Off(i1-1, 0), a.Off(0, i1-1), one, a.Off(i1-1, i1-1)); err != nil {
			panic(err)
		}

		if err = goblas.Ztrmm(Right, Upper, NoTrans, Unit, m2, m1, one, a, t.Off(i1-1, 0)); err != nil {
			panic(err)
		}

		for i = 1; i <= m2; i++ {
			for j = 1; j <= m1; j++ {
				a.Set(i+m1-1, j-1, a.Get(i+m1-1, j-1)-t.Get(i+m1-1, j-1))
				t.Set(i+m1-1, j-1, zero)
			}
		}

		//        Compute A(J1:M,J1:N) <- (Y2,R2,T2) where Q2 = I - Y2 T2 Y2^H
		if err = Zgelqt3(m2, n-m1, a.Off(i1-1, i1-1), t.Off(i1-1, i1-1)); err != nil {
			panic(err)
		}

		//        Compute T3 = T(J1:N1,1:N) = -T1 Y1^H Y2 T2
		for i = 1; i <= m2; i++ {
			for j = 1; j <= m1; j++ {
				t.Set(j-1, i+m1-1, (a.Get(j-1, i+m1-1)))
			}
		}

		if err = goblas.Ztrmm(Right, Upper, ConjTrans, Unit, m1, m2, one, a.Off(i1-1, i1-1), t.Off(0, i1-1)); err != nil {
			panic(err)
		}

		if err = goblas.Zgemm(NoTrans, ConjTrans, m1, m2, n-m, one, a.Off(0, j1-1), a.Off(i1-1, j1-1), one, t.Off(0, i1-1)); err != nil {
			panic(err)
		}

		if err = goblas.Ztrmm(Left, Upper, NoTrans, NonUnit, m1, m2, -one, t, t.Off(0, i1-1)); err != nil {
			panic(err)
		}

		if err = goblas.Ztrmm(Right, Upper, NoTrans, NonUnit, m1, m2, one, t.Off(i1-1, i1-1), t.Off(0, i1-1)); err != nil {
			panic(err)
		}

		//        Y = (Y1,Y2); L = [ L1            0  ];  T = [T1 T3]
		//                         [ A(1:N1,J1:N)  L2 ]       [ 0 T2]
	}

	return
}
