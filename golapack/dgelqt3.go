package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgelqt3 recursively computes a LQ factorization of a real M-by-N
// matrix A, using the compact WY representation of Q.
//
// Based on the algorithm of Elmroth and Gustavson,
// IBM J. Res. Develop. Vol 44 No. 4 July 2000.
func Dgelqt3(m, n *int, a *mat.Matrix, lda *int, t *mat.Matrix, ldt, info *int) {
	var one float64
	var i, i1, iinfo, j, j1, m1, m2 int
	var err error
	_ = err

	one = 1.0e+00

	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < (*m) {
		(*info) = -2
	} else if (*lda) < maxint(1, *m) {
		(*info) = -4
	} else if (*ldt) < maxint(1, *m) {
		(*info) = -6
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGELQT3"), -(*info))
		return
	}

	if (*m) == 1 {
		//        Compute Householder transform when N=1
		Dlarfg(n, a.GetIdxPtr(0), a.Vector(0, minint(2, *n)-1), lda, t.GetIdxPtr(0))

	} else {
		//        Otherwise, split A into blocks...
		m1 = (*m) / 2
		m2 = (*m) - m1
		i1 = minint(m1+1, *m)
		j1 = minint((*m)+1, *n)

		//        Compute A(1:M1,1:N) <- (Y1,R1,T1), where Q1 = I - Y1 T1 Y1^H
		Dgelqt3(&m1, n, a, lda, t, ldt, &iinfo)

		//        Compute A(J1:M,1:N) = Q1^H A(J1:M,1:N) [workspace: T(1:N1,J1:N)]
		for i = 1; i <= m2; i++ {
			for j = 1; j <= m1; j++ {
				t.Set(i+m1-1, j-1, a.Get(i+m1-1, j-1))
			}
		}
		err = goblas.Dtrmm(Right, Upper, Trans, Unit, m2, m1, one, a, *lda, t.Off(i1-1, 0), *ldt)

		err = goblas.Dgemm(NoTrans, Trans, m2, m1, (*n)-m1, one, a.Off(i1-1, i1-1), *lda, a.Off(0, i1-1), *lda, one, t.Off(i1-1, 0), *ldt)

		err = goblas.Dtrmm(Right, Upper, NoTrans, NonUnit, m2, m1, one, t, *ldt, t.Off(i1-1, 0), *ldt)

		err = goblas.Dgemm(NoTrans, NoTrans, m2, (*n)-m1, m1, -one, t.Off(i1-1, 0), *ldt, a.Off(0, i1-1), *lda, one, a.Off(i1-1, i1-1), *lda)

		err = goblas.Dtrmm(Right, Upper, NoTrans, Unit, m2, m1, one, a, *lda, t.Off(i1-1, 0), *ldt)

		for i = 1; i <= m2; i++ {
			for j = 1; j <= m1; j++ {
				a.Set(i+m1-1, j-1, a.Get(i+m1-1, j-1)-t.Get(i+m1-1, j-1))
				t.Set(i+m1-1, j-1, 0)
			}
		}

		//        Compute A(J1:M,J1:N) <- (Y2,R2,T2) where Q2 = I - Y2 T2 Y2^H
		Dgelqt3(&m2, toPtr((*n)-m1), a.Off(i1-1, i1-1), lda, t.Off(i1-1, i1-1), ldt, &iinfo)

		//        Compute T3 = T(J1:N1,1:N) = -T1 Y1^H Y2 T2
		for i = 1; i <= m2; i++ {
			for j = 1; j <= m1; j++ {
				t.Set(j-1, i+m1-1, a.Get(j-1, i+m1-1))
			}
		}

		err = goblas.Dtrmm(Right, Upper, Trans, Unit, m1, m2, one, a.Off(i1-1, i1-1), *lda, t.Off(0, i1-1), *ldt)

		err = goblas.Dgemm(NoTrans, Trans, m1, m2, (*n)-(*m), one, a.Off(0, j1-1), *lda, a.Off(i1-1, j1-1), *lda, one, t.Off(0, i1-1), *ldt)

		err = goblas.Dtrmm(Left, Upper, NoTrans, NonUnit, m1, m2, -one, t, *ldt, t.Off(0, i1-1), *ldt)

		err = goblas.Dtrmm(Right, Upper, NoTrans, NonUnit, m1, m2, one, t.Off(i1-1, i1-1), *ldt, t.Off(0, i1-1), *ldt)

		//
		//
		//        Y = (Y1,Y2); L = [ L1            0  ];  T = [T1 T3]
		//                         [ A(1:N1,J1:N)  L2 ]       [ 0 T2]
	}
}
