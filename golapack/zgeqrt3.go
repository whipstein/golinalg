package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgeqrt3 recursively computes a QR factorization of a complex M-by-N
// matrix A, using the compact WY representation of Q.
//
// Based on the algorithm of Elmroth and Gustavson,
// IBM J. Res. Develop. Vol 44 No. 4 July 2000.
func Zgeqrt3(m, n int, a, t *mat.CMatrix) (err error) {
	var one complex128
	var i, i1, j, j1, n1, n2 int

	one = (1.0e+00 + 0.0e+00*1i)

	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if m < n {
		err = fmt.Errorf("m < n: m=%v", m)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if t.Rows < max(1, n) {
		err = fmt.Errorf("t.Rows < max(1, n): t.Rows=%v, n=%v", t.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zgeqrt3", err)
		return
	}

	if n == 1 {
		//        Compute Householder transform when N=1
		*a.GetPtr(0, 0), *t.GetPtr(0, 0) = Zlarfg(m, a.Get(0, 0), a.Off(min(2, m)-1, 0).CVector(), 1)

	} else {
		//        Otherwise, split A into blocks...
		n1 = n / 2
		n2 = n - n1
		j1 = min(n1+1, n)
		i1 = min(n+1, m)

		//        Compute A(1:M,1:N1) <- (Y1,R1,T1), where Q1 = I - Y1 T1 Y1^H
		if err = Zgeqrt3(m, n1, a, t); err != nil {
			panic(err)
		}

		//        Compute A(1:M,J1:N) = Q1^H A(1:M,J1:N) [workspace: T(1:N1,J1:N)]
		for j = 1; j <= n2; j++ {
			for i = 1; i <= n1; i++ {
				t.Set(i-1, j+n1-1, a.Get(i-1, j+n1-1))
			}
		}
		if err = t.Off(0, j1-1).Trmm(Left, Lower, ConjTrans, Unit, n1, n2, one, a); err != nil {
			panic(err)
		}

		if err = t.Off(0, j1-1).Gemm(ConjTrans, NoTrans, n1, n2, m-n1, one, a.Off(j1-1, 0), a.Off(j1-1, j1-1), one); err != nil {
			panic(err)
		}

		if err = t.Off(0, j1-1).Trmm(Left, Upper, ConjTrans, NonUnit, n1, n2, one, t); err != nil {
			panic(err)
		}

		if err = a.Off(j1-1, j1-1).Gemm(NoTrans, NoTrans, m-n1, n2, n1, -one, a.Off(j1-1, 0), t.Off(0, j1-1), one); err != nil {
			panic(err)
		}

		if err = t.Off(0, j1-1).Trmm(Left, Lower, NoTrans, Unit, n1, n2, one, a); err != nil {
			panic(err)
		}

		for j = 1; j <= n2; j++ {
			for i = 1; i <= n1; i++ {
				a.Set(i-1, j+n1-1, a.Get(i-1, j+n1-1)-t.Get(i-1, j+n1-1))
			}
		}

		//        Compute A(J1:M,J1:N) <- (Y2,R2,T2) where Q2 = I - Y2 T2 Y2^H
		if err = Zgeqrt3(m-n1, n2, a.Off(j1-1, j1-1), t.Off(j1-1, j1-1)); err != nil {
			panic(err)
		}

		//        Compute T3 = T(1:N1,J1:N) = -T1 Y1^H Y2 T2
		for i = 1; i <= n1; i++ {
			for j = 1; j <= n2; j++ {
				t.Set(i-1, j+n1-1, a.GetConj(j+n1-1, i-1))
			}
		}

		if err = t.Off(0, j1-1).Trmm(Right, Lower, NoTrans, Unit, n1, n2, one, a.Off(j1-1, j1-1)); err != nil {
			panic(err)
		}

		if err = t.Off(0, j1-1).Gemm(ConjTrans, NoTrans, n1, n2, m-n, one, a.Off(i1-1, 0), a.Off(i1-1, j1-1), one); err != nil {
			panic(err)
		}

		if err = t.Off(0, j1-1).Trmm(Left, Upper, NoTrans, NonUnit, n1, n2, -one, t); err != nil {
			panic(err)
		}

		if err = t.Off(0, j1-1).Trmm(Right, Upper, NoTrans, NonUnit, n1, n2, one, t.Off(j1-1, j1-1)); err != nil {
			panic(err)
		}

		//        Y = (Y1,Y2); R = [ R1  A(1:N1,J1:N) ];  T = [T1 T3]
		//                         [  0        R2     ]       [ 0 T2]
	}

	return
}
