package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dorm22 overwrites the general real M-by-N matrix C with
//
//                  SIDE = 'L'     SIDE = 'R'
//  TRANS = 'N':      Q * C          C * Q
//  TRANS = 'T':      Q**T * C       C * Q**T
//
//  where Q is a real orthogonal matrix of order NQ, with NQ = M if
//  SIDE = 'L' and NQ = N if SIDE = 'R'.
//  The orthogonal matrix Q processes a 2-by-2 block structure
//
//         [  Q11  Q12  ]
//     Q = [            ]
//         [  Q21  Q22  ],
//
//  where Q12 is an N1-by-N1 lower triangular matrix and Q21 is an
//  N2-by-N2 upper triangular matrix.
func Dorm22(side, trans byte, m, n, n1, n2 *int, q *mat.Matrix, ldq *int, c *mat.Matrix, ldc *int, work *mat.Vector, lwork, info *int) {
	var left, lquery, notran bool
	var one float64
	var i, ldwork, len, lwkopt, nb, nq, nw int

	one = 1.0

	//     Test the input arguments
	(*info) = 0
	left = side == 'L'
	notran = trans == 'N'
	lquery = ((*lwork) == -1)

	//     NQ is the order of Q;
	//     NW is the minimum dimension of WORK.
	if left {
		nq = (*m)
	} else {
		nq = (*n)
	}
	nw = nq
	if (*n1) == 0 || (*n2) == 0 {
		nw = 1
	}
	if !left && side != 'R' {
		(*info) = -1
	} else if trans != 'N' && trans != 'T' {
		(*info) = -2
	} else if (*m) < 0 {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*n1) < 0 || (*n1)+(*n2) != nq {
		(*info) = -5
	} else if (*n2) < 0 {
		(*info) = -6
	} else if (*ldq) < maxint(1, nq) {
		(*info) = -8
	} else if (*ldc) < maxint(1, *m) {
		(*info) = -10
	} else if (*lwork) < nw && !lquery {
		(*info) = -12
	}

	if (*info) == 0 {
		lwkopt = (*m) * (*n)
		work.Set(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DORM22"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		work.Set(0, 1)
		return
	}

	//     Degenerate cases (N1 = 0 or N2 = 0) are handled using DTRMM.
	if (*n1) == 0 {
		goblas.Dtrmm(mat.SideByte(side), Upper, mat.TransByte(trans), NonUnit, m, n, &one, q, ldq, c, ldc)
		work.Set(0, one)
		return
	} else if (*n2) == 0 {
		goblas.Dtrmm(mat.SideByte(side), Lower, mat.TransByte(trans), NonUnit, m, n, &one, q, ldq, c, ldc)
		work.Set(0, one)
		return
	}

	//     Compute the largest chunk size available from the workspace.
	nb = maxint(1, minint(*lwork, lwkopt)/nq)

	if left {
		if notran {
			for i = 1; i <= (*n); i += nb {
				len = minint(nb, (*n)-i+1)
				ldwork = (*m)

				//              Multiply bottom part of C by Q12.
				Dlacpy('A', n1, &len, c.Off((*n2)+1-1, i-1), ldc, work.Matrix(ldwork, opts), &ldwork)
				goblas.Dtrmm(Left, Lower, NoTrans, NonUnit, n1, &len, &one, q.Off(0, (*n2)+1-1), ldq, work.Matrix(ldwork, opts), &ldwork)

				//              Multiply top part of C by Q11.
				goblas.Dgemm(NoTrans, NoTrans, n1, &len, n2, &one, q, ldq, c.Off(0, i-1), ldc, &one, work.Matrix(ldwork, opts), &ldwork)

				//              Multiply top part of C by Q21.
				Dlacpy('A', n2, &len, c.Off(0, i-1), ldc, work.MatrixOff((*n1)+1-1, ldwork, opts), &ldwork)
				goblas.Dtrmm(Left, Upper, NoTrans, NonUnit, n2, &len, &one, q.Off((*n1)+1-1, 0), ldq, work.MatrixOff((*n1)+1-1, ldwork, opts), &ldwork)

				//              Multiply bottom part of C by Q22.
				goblas.Dgemm(NoTrans, NoTrans, n2, &len, n1, &one, q.Off((*n1)+1-1, (*n2)+1-1), ldq, c.Off((*n2)+1-1, i-1), ldc, &one, work.MatrixOff((*n1)+1-1, ldwork, opts), &ldwork)

				//              Copy everything back.
				Dlacpy('A', m, &len, work.Matrix(ldwork, opts), &ldwork, c.Off(0, i-1), ldc)
			}
		} else {
			for i = 1; i <= (*n); i += nb {
				len = minint(nb, (*n)-i+1)
				ldwork = (*m)

				//              Multiply bottom part of C by Q21**T.
				Dlacpy('A', n2, &len, c.Off((*n1)+1-1, i-1), ldc, work.Matrix(ldwork, opts), &ldwork)
				goblas.Dtrmm(Left, Upper, Trans, NonUnit, n2, &len, &one, q.Off((*n1)+1-1, 0), ldq, work.Matrix(ldwork, opts), &ldwork)

				//              Multiply top part of C by Q11**T.
				goblas.Dgemm(Trans, NoTrans, n2, &len, n1, &one, q, ldq, c.Off(0, i-1), ldc, &one, work.Matrix(ldwork, opts), &ldwork)

				//              Multiply top part of C by Q12**T.
				Dlacpy('A', n1, &len, c.Off(0, i-1), ldc, work.MatrixOff((*n2)+1-1, ldwork, opts), &ldwork)
				goblas.Dtrmm(Left, Lower, Trans, NonUnit, n1, &len, &one, q.Off(0, (*n2)+1-1), ldq, work.MatrixOff((*n2)+1-1, ldwork, opts), &ldwork)

				//              Multiply bottom part of C by Q22**T.
				goblas.Dgemm(Trans, NoTrans, n1, &len, n2, &one, q.Off((*n1)+1-1, (*n2)+1-1), ldq, c.Off((*n1)+1-1, i-1), ldc, &one, work.MatrixOff((*n2)+1-1, ldwork, opts), &ldwork)

				//              Copy everything back.
				Dlacpy('A', m, &len, work.Matrix(ldwork, opts), &ldwork, c.Off(0, i-1), ldc)
			}
		}
	} else {
		if notran {
			for i = 1; i <= (*m); i += nb {
				len = minint(nb, (*m)-i+1)
				ldwork = len

				//              Multiply right part of C by Q21.
				Dlacpy('A', &len, n2, c.Off(i-1, (*n1)+1-1), ldc, work.Matrix(ldwork, opts), &ldwork)
				goblas.Dtrmm(Right, Upper, NoTrans, NonUnit, &len, n2, &one, q.Off((*n1)+1-1, 0), ldq, work.Matrix(ldwork, opts), &ldwork)

				//              Multiply left part of C by Q11.
				goblas.Dgemm(NoTrans, NoTrans, &len, n2, n1, &one, c.Off(i-1, 0), ldc, q, ldq, &one, work.Matrix(ldwork, opts), &ldwork)

				//              Multiply left part of C by Q12.
				Dlacpy('A', &len, n1, c.Off(i-1, 0), ldc, work.MatrixOff(1+(*n2)*ldwork-1, ldwork, opts), &ldwork)
				goblas.Dtrmm(Right, Lower, NoTrans, NonUnit, &len, n1, &one, q.Off(0, (*n2)+1-1), ldq, work.MatrixOff(1+(*n2)*ldwork-1, ldwork, opts), &ldwork)

				//              Multiply right part of C by Q22.
				goblas.Dgemm(NoTrans, NoTrans, &len, n1, n2, &one, c.Off(i-1, (*n1)+1-1), ldc, q.Off((*n1)+1-1, (*n2)+1-1), ldq, &one, work.MatrixOff(1+(*n2)*ldwork-1, ldwork, opts), &ldwork)

				//              Copy everything back.
				Dlacpy('A', &len, n, work.Matrix(ldwork, opts), &ldwork, c.Off(i-1, 0), ldc)
			}
		} else {
			for i = 1; i <= (*m); i += nb {
				len = minint(nb, (*m)-i+1)
				ldwork = len

				//              Multiply right part of C by Q12**T.
				Dlacpy('A', &len, n1, c.Off(i-1, (*n2)+1-1), ldc, work.Matrix(ldwork, opts), &ldwork)
				goblas.Dtrmm(Right, Lower, Trans, NonUnit, &len, n1, &one, q.Off(0, (*n2)+1-1), ldq, work.Matrix(ldwork, opts), &ldwork)

				//              Multiply left part of C by Q11**T.
				goblas.Dgemm(NoTrans, Trans, &len, n1, n2, &one, c.Off(i-1, 0), ldc, q, ldq, &one, work.Matrix(ldwork, opts), &ldwork)

				//              Multiply left part of C by Q21**T.
				Dlacpy('A', &len, n2, c.Off(i-1, 0), ldc, work.MatrixOff(1+(*n1)*ldwork-1, ldwork, opts), &ldwork)
				goblas.Dtrmm(Right, Upper, Trans, NonUnit, &len, n2, &one, q.Off((*n1)+1-1, 0), ldq, work.MatrixOff(1+(*n1)*ldwork-1, ldwork, opts), &ldwork)

				//              Multiply right part of C by Q22**T.
				goblas.Dgemm(NoTrans, Trans, &len, n2, n1, &one, c.Off(i-1, (*n2)+1-1), ldc, q.Off((*n1)+1-1, (*n2)+1-1), ldq, &one, work.MatrixOff(1+(*n1)*ldwork-1, ldwork, opts), &ldwork)

				//              Copy everything back.
				Dlacpy('A', &len, n, work.Matrix(ldwork, opts), &ldwork, c.Off(i-1, 0), ldc)
			}
		}
	}

	work.Set(0, float64(lwkopt))
}
