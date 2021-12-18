package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgelqt3 recursively computes a LQ factorization of a real M-by-N
// matrix A, using the compact WY representation of Q.
//
// Based on the algorithm of Elmroth and Gustavson,
// IBM J. Res. Develop. Vol 44 No. 4 July 2000.
func Dgelqt3(m, n int, a, t *mat.Matrix) (err error) {
	var one float64
	var i, i1, j, j1, m1, m2 int

	one = 1.0e+00

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
		gltest.Xerbla2("Dgelqt3", err)
		return
	}

	if m == 1 {
		//        Compute Householder transform when N=1
		*a.GetIdxPtr(0), *t.GetIdxPtr(0) = Dlarfg(n, a.GetIdx(0), a.Off(0, min(2, n)-1).Vector(), a.Rows)

	} else {
		//        Otherwise, split A into blocks...
		m1 = m / 2
		m2 = m - m1
		i1 = min(m1+1, m)
		j1 = min(m+1, n)

		//        Compute A(1:M1,1:N) <- (Y1,R1,T1), where Q1 = I - Y1 T1 Y1^H
		if err = Dgelqt3(m1, n, a, t); err != nil {
			panic(err)
		}

		//        Compute A(J1:M,1:N) = Q1^H A(J1:M,1:N) [workspace: T(1:N1,J1:N)]
		for i = 1; i <= m2; i++ {
			for j = 1; j <= m1; j++ {
				t.Set(i+m1-1, j-1, a.Get(i+m1-1, j-1))
			}
		}
		if err = t.Off(i1-1, 0).Trmm(Right, Upper, Trans, Unit, m2, m1, one, a); err != nil {
			panic(err)
		}

		if err = t.Off(i1-1, 0).Gemm(NoTrans, Trans, m2, m1, n-m1, one, a.Off(i1-1, i1-1), a.Off(0, i1-1), one); err != nil {
			panic(err)
		}

		if err = t.Off(i1-1, 0).Trmm(Right, Upper, NoTrans, NonUnit, m2, m1, one, t); err != nil {
			panic(err)
		}

		if err = a.Off(i1-1, i1-1).Gemm(NoTrans, NoTrans, m2, n-m1, m1, -one, t.Off(i1-1, 0), a.Off(0, i1-1), one); err != nil {
			panic(err)
		}

		if err = t.Off(i1-1, 0).Trmm(Right, Upper, NoTrans, Unit, m2, m1, one, a); err != nil {
			panic(err)
		}

		for i = 1; i <= m2; i++ {
			for j = 1; j <= m1; j++ {
				a.Set(i+m1-1, j-1, a.Get(i+m1-1, j-1)-t.Get(i+m1-1, j-1))
				t.Set(i+m1-1, j-1, 0)
			}
		}

		//        Compute A(J1:M,J1:N) <- (Y2,R2,T2) where Q2 = I - Y2 T2 Y2^H
		if err = Dgelqt3(m2, n-m1, a.Off(i1-1, i1-1), t.Off(i1-1, i1-1)); err != nil {
			panic(err)
		}

		//        Compute T3 = T(J1:N1,1:N) = -T1 Y1^H Y2 T2
		for i = 1; i <= m2; i++ {
			for j = 1; j <= m1; j++ {
				t.Set(j-1, i+m1-1, a.Get(j-1, i+m1-1))
			}
		}

		if err = t.Off(0, i1-1).Trmm(Right, Upper, Trans, Unit, m1, m2, one, a.Off(i1-1, i1-1)); err != nil {
			panic(err)
		}

		if err = t.Off(0, i1-1).Gemm(NoTrans, Trans, m1, m2, n-m, one, a.Off(0, j1-1), a.Off(i1-1, j1-1), one); err != nil {
			panic(err)
		}

		if err = t.Off(0, i1-1).Trmm(Left, Upper, NoTrans, NonUnit, m1, m2, -one, t); err != nil {
			panic(err)
		}

		if err = t.Off(0, i1-1).Trmm(Right, Upper, NoTrans, NonUnit, m1, m2, one, t.Off(i1-1, i1-1)); err != nil {
			panic(err)
		}

		//
		//
		//        Y = (Y1,Y2); L = [ L1            0  ];  T = [T1 T3]
		//                         [ A(1:N1,J1:N)  L2 ]       [ 0 T2]
	}

	return
}
