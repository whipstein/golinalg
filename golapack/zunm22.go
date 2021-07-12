package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zunm22 overwrites the general complex M-by-N matrix C with
//
//                  SIDE = 'L'     SIDE = 'R'
//  TRANS = 'N':      Q * C          C * Q
//  TRANS = 'C':      Q**H * C       C * Q**H
//
//  where Q is a complex unitary matrix of order NQ, with NQ = M if
//  SIDE = 'L' and NQ = N if SIDE = 'R'.
//  The unitary matrix Q processes a 2-by-2 block structure
//
//         [  Q11  Q12  ]
//     Q = [            ]
//         [  Q21  Q22  ],
//
//  where Q12 is an N1-by-N1 lower triangular matrix and Q21 is an
//  N2-by-N2 upper triangular matrix.
func Zunm22(side, trans byte, m, n, n1, n2 *int, q *mat.CMatrix, ldq *int, c *mat.CMatrix, ldc *int, work *mat.CVector, lwork, info *int) {
	var left, lquery, notran bool
	var one complex128
	var i, ldwork, len, lwkopt, nb, nq, nw int
	var err error
	_ = err

	one = (1.0 + 0.0*1i)

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
	} else if trans != 'N' && trans != 'C' {
		(*info) = -2
	} else if (*m) < 0 {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*n1) < 0 || (*n1)+(*n2) != nq {
		(*info) = -5
	} else if (*n2) < 0 {
		(*info) = -6
	} else if (*ldq) < max(1, nq) {
		(*info) = -8
	} else if (*ldc) < max(1, *m) {
		(*info) = -10
	} else if (*lwork) < nw && !lquery {
		(*info) = -12
	}

	if (*info) == 0 {
		lwkopt = (*m) * (*n)
		work.SetRe(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNM22"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		work.Set(0, 1)
		return
	}

	//     Degenerate cases (N1 = 0 or N2 = 0) are handled using ZTRMM.
	if (*n1) == 0 {
		err = goblas.Ztrmm(mat.SideByte(side), Upper, mat.TransByte(trans), NonUnit, *m, *n, one, q, c)
		work.Set(0, one)
		return
	} else if (*n2) == 0 {
		err = goblas.Ztrmm(mat.SideByte(side), Lower, mat.TransByte(trans), NonUnit, *m, *n, one, q, c)
		work.Set(0, one)
		return
	}

	//     Compute the largest chunk size available from the workspace.
	nb = max(1, min(*lwork, lwkopt)/nq)

	if left {
		if notran {
			for i = 1; i <= (*n); i += nb {
				len = min(nb, (*n)-i+1)
				ldwork = (*m)

				//              Multiply bottom part of C by Q12.
				Zlacpy('A', n1, &len, c.Off((*n2), i-1), ldc, work.CMatrix(ldwork, opts), &ldwork)
				err = goblas.Ztrmm(Left, Lower, NoTrans, NonUnit, *n1, len, one, q.Off(0, (*n2)), work.CMatrix(ldwork, opts))

				//              Multiply top part of C by Q11.
				err = goblas.Zgemm(NoTrans, NoTrans, *n1, len, *n2, one, q, c.Off(0, i-1), one, work.CMatrix(ldwork, opts))

				//              Multiply top part of C by Q21.
				Zlacpy('A', n2, &len, c.Off(0, i-1), ldc, work.CMatrixOff((*n1), ldwork, opts), &ldwork)
				err = goblas.Ztrmm(Left, Upper, NoTrans, NonUnit, *n2, len, one, q.Off((*n1), 0), work.CMatrixOff((*n1), ldwork, opts))

				//              Multiply bottom part of C by Q22.
				err = goblas.Zgemm(NoTrans, NoTrans, *n2, len, *n1, one, q.Off((*n1), (*n2)), c.Off((*n2), i-1), one, work.CMatrixOff((*n1), ldwork, opts))

				//              Copy everything back.
				Zlacpy('A', m, &len, work.CMatrix(ldwork, opts), &ldwork, c.Off(0, i-1), ldc)
			}
		} else {
			for i = 1; i <= (*n); i += nb {
				len = min(nb, (*n)-i+1)
				ldwork = (*m)

				//              Multiply bottom part of C by Q21**H.
				Zlacpy('A', n2, &len, c.Off((*n1), i-1), ldc, work.CMatrix(ldwork, opts), &ldwork)
				err = goblas.Ztrmm(Left, Upper, ConjTrans, NonUnit, *n2, len, one, q.Off((*n1), 0), work.CMatrix(ldwork, opts))

				//              Multiply top part of C by Q11**H.
				err = goblas.Zgemm(ConjTrans, NoTrans, *n2, len, *n1, one, q, c.Off(0, i-1), one, work.CMatrix(ldwork, opts))

				//              Multiply top part of C by Q12**H.
				Zlacpy('A', n1, &len, c.Off(0, i-1), ldc, work.CMatrixOff((*n2), ldwork, opts), &ldwork)
				err = goblas.Ztrmm(Left, Lower, ConjTrans, NonUnit, *n1, len, one, q.Off(0, (*n2)), work.CMatrixOff((*n2), ldwork, opts))

				//              Multiply bottom part of C by Q22**H.
				err = goblas.Zgemm(ConjTrans, NoTrans, *n1, len, *n2, one, q.Off((*n1), (*n2)), c.Off((*n1), i-1), one, work.CMatrixOff((*n2), ldwork, opts))

				//              Copy everything back.
				Zlacpy('A', m, &len, work.CMatrix(ldwork, opts), &ldwork, c.Off(0, i-1), ldc)
			}
		}
	} else {
		if notran {
			for i = 1; i <= (*m); i += nb {
				len = min(nb, (*m)-i+1)
				ldwork = len

				//              Multiply right part of C by Q21.
				Zlacpy('A', &len, n2, c.Off(i-1, (*n1)), ldc, work.CMatrix(ldwork, opts), &ldwork)
				err = goblas.Ztrmm(Right, Upper, NoTrans, NonUnit, len, *n2, one, q.Off((*n1), 0), work.CMatrix(ldwork, opts))

				//              Multiply left part of C by Q11.
				err = goblas.Zgemm(NoTrans, NoTrans, len, *n2, *n1, one, c.Off(i-1, 0), q, one, work.CMatrix(ldwork, opts))

				//              Multiply left part of C by Q12.
				Zlacpy('A', &len, n1, c.Off(i-1, 0), ldc, work.CMatrixOff(1+(*n2)*ldwork-1, ldwork, opts), &ldwork)
				err = goblas.Ztrmm(Right, Lower, NoTrans, NonUnit, len, *n1, one, q.Off(0, (*n2)), work.CMatrixOff(1+(*n2)*ldwork-1, ldwork, opts))

				//              Multiply right part of C by Q22.
				err = goblas.Zgemm(NoTrans, NoTrans, len, *n1, *n2, one, c.Off(i-1, (*n1)), q.Off((*n1), (*n2)), one, work.CMatrixOff(1+(*n2)*ldwork-1, ldwork, opts))

				//              Copy everything back.
				Zlacpy('A', &len, n, work.CMatrix(ldwork, opts), &ldwork, c.Off(i-1, 0), ldc)
			}
		} else {
			for i = 1; i <= (*m); i += nb {
				len = min(nb, (*m)-i+1)
				ldwork = len

				//              Multiply right part of C by Q12**H.
				Zlacpy('A', &len, n1, c.Off(i-1, (*n2)), ldc, work.CMatrix(ldwork, opts), &ldwork)
				err = goblas.Ztrmm(Right, Lower, ConjTrans, NonUnit, len, *n1, one, q.Off(0, (*n2)), work.CMatrix(ldwork, opts))

				//              Multiply left part of C by Q11**H.
				err = goblas.Zgemm(NoTrans, ConjTrans, len, *n1, *n2, one, c.Off(i-1, 0), q, one, work.CMatrix(ldwork, opts))

				//              Multiply left part of C by Q21**H.
				Zlacpy('A', &len, n2, c.Off(i-1, 0), ldc, work.CMatrixOff(1+(*n1)*ldwork-1, ldwork, opts), &ldwork)
				err = goblas.Ztrmm(Right, Upper, ConjTrans, NonUnit, len, *n2, one, q.Off((*n1), 0), work.CMatrixOff(1+(*n1)*ldwork-1, ldwork, opts))

				//              Multiply right part of C by Q22**H.
				err = goblas.Zgemm(NoTrans, ConjTrans, len, *n2, *n1, one, c.Off(i-1, (*n2)), q.Off((*n1), (*n2)), one, work.CMatrixOff(1+(*n1)*ldwork-1, ldwork, opts))

				//              Copy everything back.
				Zlacpy('A', &len, n, work.CMatrix(ldwork, opts), &ldwork, c.Off(i-1, 0), ldc)
			}
		}
	}

	work.SetRe(0, float64(lwkopt))
}
