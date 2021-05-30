package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dgeqrt3 recursively computes a QR factorization of a real M-by-N
// matrix A, using the compact WY representation of Q.
//
// Based on the algorithm of Elmroth and Gustavson,
// IBM J. Res. Develop. Vol 44 No. 4 July 2000.
func Dgeqrt3(m, n *int, a *mat.Matrix, lda *int, t *mat.Matrix, ldt, info *int) {
	var one float64
	var i, i1, iinfo, j, j1, n1, n2 int

	one = 1.0e+00

	(*info) = 0
	if (*n) < 0 {
		(*info) = -2
	} else if (*m) < (*n) {
		(*info) = -1
	} else if (*lda) < maxint(1, *m) {
		(*info) = -4
	} else if (*ldt) < maxint(1, *n) {
		(*info) = -6
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGEQRT3"), -(*info))
		return
	}

	if (*n) == 1 {
		//        Compute Householder transform when N=1
		Dlarfg(m, a.GetPtr(0, 0), a.Vector(minint(2, *m)-1, 0), func() *int { y := 1; return &y }(), t.GetPtr(0, 0))

	} else {
		//        Otherwise, split A into blocks...
		n1 = (*n) / 2
		n2 = (*n) - n1
		j1 = minint(n1+1, *n)
		i1 = minint((*n)+1, *m)

		//        Compute A(1:M,1:N1) <- (Y1,R1,T1), where Q1 = I - Y1 T1 Y1^H
		Dgeqrt3(m, &n1, a, lda, t, ldt, &iinfo)

		//        Compute A(1:M,J1:N) = Q1^H A(1:M,J1:N) [workspace: T(1:N1,J1:N)]
		for j = 1; j <= n2; j++ {
			for i = 1; i <= n1; i++ {
				t.Set(i-1, j+n1-1, a.Get(i-1, j+n1-1))
			}
		}
		goblas.Dtrmm(Left, Lower, Trans, Unit, &n1, &n2, &one, a, lda, t.Off(0, j1-1), ldt)

		goblas.Dgemm(Trans, NoTrans, &n1, &n2, toPtr((*m)-n1), &one, a.Off(j1-1, 0), lda, a.Off(j1-1, j1-1), lda, &one, t.Off(0, j1-1), ldt)

		goblas.Dtrmm(Left, Upper, Trans, NonUnit, &n1, &n2, &one, t, ldt, t.Off(0, j1-1), ldt)

		goblas.Dgemm(NoTrans, NoTrans, toPtr((*m)-n1), &n2, &n1, toPtrf64(-one), a.Off(j1-1, 0), lda, t.Off(0, j1-1), ldt, &one, a.Off(j1-1, j1-1), lda)

		goblas.Dtrmm(Left, Lower, NoTrans, Unit, &n1, &n2, &one, a, lda, t.Off(0, j1-1), ldt)

		for j = 1; j <= n2; j++ {
			for i = 1; i <= n1; i++ {
				a.Set(i-1, j+n1-1, a.Get(i-1, j+n1-1)-t.Get(i-1, j+n1-1))
			}
		}

		//        Compute A(J1:M,J1:N) <- (Y2,R2,T2) where Q2 = I - Y2 T2 Y2^H
		Dgeqrt3(toPtr((*m)-n1), &n2, a.Off(j1-1, j1-1), lda, t.Off(j1-1, j1-1), ldt, &iinfo)

		//        Compute T3 = T(1:N1,J1:N) = -T1 Y1^H Y2 T2
		for i = 1; i <= n1; i++ {
			for j = 1; j <= n2; j++ {
				t.Set(i-1, j+n1-1, a.Get(j+n1-1, i-1))
			}
		}

		goblas.Dtrmm(Right, Lower, NoTrans, Unit, &n1, &n2, &one, a.Off(j1-1, j1-1), lda, t.Off(0, j1-1), ldt)

		goblas.Dgemm(Trans, NoTrans, &n1, &n2, toPtr((*m)-(*n)), &one, a.Off(i1-1, 0), lda, a.Off(i1-1, j1-1), lda, &one, t.Off(0, j1-1), ldt)

		goblas.Dtrmm(Left, Upper, NoTrans, NonUnit, &n1, &n2, toPtrf64(-one), t, ldt, t.Off(0, j1-1), ldt)

		goblas.Dtrmm(Right, Upper, NoTrans, NonUnit, &n1, &n2, &one, t.Off(j1-1, j1-1), ldt, t.Off(0, j1-1), ldt)

		//        Y = (Y1,Y2); R = [ R1  A(1:N1,J1:N) ];  T = [T1 T3]
		//                         [  0        R2     ]       [ 0 T2]
	}
}
