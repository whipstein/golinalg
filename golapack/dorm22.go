package golapack

import (
	"fmt"

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
func Dorm22(side mat.MatSide, trans mat.MatTrans, m, n, n1, n2 int, q, c *mat.Matrix, work *mat.Vector, lwork int) (err error) {
	var left, lquery, notran bool
	var one float64
	var i, ldwork, len, lwkopt, nb, nq, nw int

	one = 1.0

	//     Test the input arguments
	left = side == Left
	notran = trans == NoTrans
	lquery = (lwork == -1)

	//     NQ is the order of Q;
	//     NW is the minimum dimension of WORK.
	if left {
		nq = m
	} else {
		nq = n
	}
	nw = nq
	if n1 == 0 || n2 == 0 {
		nw = 1
	}
	if !left && side != Right {
		err = fmt.Errorf("!left && side != Right: side=%s", side)
	} else if trans != NoTrans && trans != Trans {
		err = fmt.Errorf("trans != NoTrans && trans != Trans: trans=%s", trans)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if n1 < 0 || n1+n2 != nq {
		err = fmt.Errorf("n1 < 0 || n1+n2 != nq: n1=%v, n2=%v, nq=%v", n1, n2, nq)
	} else if n2 < 0 {
		err = fmt.Errorf("n2 < 0: n2=%v", n2)
	} else if q.Rows < max(1, nq) {
		err = fmt.Errorf("q.Rows < max(1, nq): q.Rows=%v, nq=%v", q.Rows, nq)
	} else if c.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows < max(1, m): c.Rows=%v, m=%v", c.Rows, m)
	} else if lwork < nw && !lquery {
		err = fmt.Errorf("lwork < nw && !lquery: lwork=%v, nw=%v, lquery=%v", lwork, nw, lquery)
	}

	if err == nil {
		lwkopt = m * n
		work.Set(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("Dorm22", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 {
		work.Set(0, 1)
		return
	}

	//     Degenerate cases (N1 = 0 or N2 = 0) are handled using DTRMM.
	if n1 == 0 {
		if err = c.Trmm(side, Upper, trans, NonUnit, m, n, one, q); err != nil {
			panic(err)
		}
		work.Set(0, one)
		return
	} else if n2 == 0 {
		if err = c.Trmm(side, Lower, trans, NonUnit, m, n, one, q); err != nil {
			panic(err)
		}
		work.Set(0, one)
		return
	}

	//     Compute the largest chunk size available from the workspace.
	nb = max(1, min(lwork, lwkopt)/nq)

	if left {
		if notran {
			for i = 1; i <= n; i += nb {
				len = min(nb, n-i+1)
				ldwork = m

				//              Multiply bottom part of C by Q12.
				Dlacpy(Full, n1, len, c.Off(n2, i-1), work.Matrix(ldwork, opts))
				if err = work.Matrix(ldwork, opts).Trmm(Left, Lower, NoTrans, NonUnit, n1, len, one, q.Off(0, n2)); err != nil {
					panic(err)
				}

				//              Multiply top part of C by Q11.
				if err = work.Matrix(ldwork, opts).Gemm(NoTrans, NoTrans, n1, len, n2, one, q, c.Off(0, i-1), one); err != nil {
					panic(err)
				}

				//              Multiply top part of C by Q21.
				Dlacpy(Full, n2, len, c.Off(0, i-1), work.Off(n1).Matrix(ldwork, opts))
				if err = work.Off(n1).Matrix(ldwork, opts).Trmm(Left, Upper, NoTrans, NonUnit, n2, len, one, q.Off(n1, 0)); err != nil {
					panic(err)
				}

				//              Multiply bottom part of C by Q22.
				if err = work.Off(n1).Matrix(ldwork, opts).Gemm(NoTrans, NoTrans, n2, len, n1, one, q.Off(n1, n2), c.Off(n2, i-1), one); err != nil {
					panic(err)
				}

				//              Copy everything back.
				Dlacpy(Full, m, len, work.Matrix(ldwork, opts), c.Off(0, i-1))
			}
		} else {
			for i = 1; i <= n; i += nb {
				len = min(nb, n-i+1)
				ldwork = m

				//              Multiply bottom part of C by Q21**T.
				Dlacpy(Full, n2, len, c.Off(n1, i-1), work.Matrix(ldwork, opts))
				if err = work.Matrix(ldwork, opts).Trmm(Left, Upper, Trans, NonUnit, n2, len, one, q.Off(n1, 0)); err != nil {
					panic(err)
				}

				//              Multiply top part of C by Q11**T.
				if err = work.Matrix(ldwork, opts).Gemm(Trans, NoTrans, n2, len, n1, one, q, c.Off(0, i-1), one); err != nil {
					panic(err)
				}

				//              Multiply top part of C by Q12**T.
				Dlacpy(Full, n1, len, c.Off(0, i-1), work.Off(n2).Matrix(ldwork, opts))
				if err = work.Off(n2).Matrix(ldwork, opts).Trmm(Left, Lower, Trans, NonUnit, n1, len, one, q.Off(0, n2)); err != nil {
					panic(err)
				}

				//              Multiply bottom part of C by Q22**T.
				if err = work.Off(n2).Matrix(ldwork, opts).Gemm(Trans, NoTrans, n1, len, n2, one, q.Off(n1, n2), c.Off(n1, i-1), one); err != nil {
					panic(err)
				}

				//              Copy everything back.
				Dlacpy(Full, m, len, work.Matrix(ldwork, opts), c.Off(0, i-1))
			}
		}
	} else {
		if notran {
			for i = 1; i <= m; i += nb {
				len = min(nb, m-i+1)
				ldwork = len

				//              Multiply right part of C by Q21.
				Dlacpy(Full, len, n2, c.Off(i-1, n1), work.Matrix(ldwork, opts))
				if err = work.Matrix(ldwork, opts).Trmm(Right, Upper, NoTrans, NonUnit, len, n2, one, q.Off(n1, 0)); err != nil {
					panic(err)
				}

				//              Multiply left part of C by Q11.
				if err = work.Matrix(ldwork, opts).Gemm(NoTrans, NoTrans, len, n2, n1, one, c.Off(i-1, 0), q, one); err != nil {
					panic(err)
				}

				//              Multiply left part of C by Q12.
				Dlacpy(Full, len, n1, c.Off(i-1, 0), work.Off(1+n2*ldwork-1).Matrix(ldwork, opts))
				if err = work.Off(1+n2*ldwork-1).Matrix(ldwork, opts).Trmm(Right, Lower, NoTrans, NonUnit, len, n1, one, q.Off(0, n2)); err != nil {
					panic(err)
				}

				//              Multiply right part of C by Q22.
				if err = work.Off(1+n2*ldwork-1).Matrix(ldwork, opts).Gemm(NoTrans, NoTrans, len, n1, n2, one, c.Off(i-1, n1), q.Off(n1, n2), one); err != nil {
					panic(err)
				}

				//              Copy everything back.
				Dlacpy(Full, len, n, work.Matrix(ldwork, opts), c.Off(i-1, 0))
			}
		} else {
			for i = 1; i <= m; i += nb {
				len = min(nb, m-i+1)
				ldwork = len

				//              Multiply right part of C by Q12**T.
				Dlacpy(Full, len, n1, c.Off(i-1, n2), work.Matrix(ldwork, opts))
				if err = work.Matrix(ldwork, opts).Trmm(Right, Lower, Trans, NonUnit, len, n1, one, q.Off(0, n2)); err != nil {
					panic(err)
				}

				//              Multiply left part of C by Q11**T.
				if err = work.Matrix(ldwork, opts).Gemm(NoTrans, Trans, len, n1, n2, one, c.Off(i-1, 0), q, one); err != nil {
					panic(err)
				}

				//              Multiply left part of C by Q21**T.
				Dlacpy(Full, len, n2, c.Off(i-1, 0), work.Off(1+n1*ldwork-1).Matrix(ldwork, opts))
				if err = work.Off(1+n1*ldwork-1).Matrix(ldwork, opts).Trmm(Right, Upper, Trans, NonUnit, len, n2, one, q.Off(n1, 0)); err != nil {
					panic(err)
				}

				//              Multiply right part of C by Q22**T.
				if err = work.Off(1+n1*ldwork-1).Matrix(ldwork, opts).Gemm(NoTrans, Trans, len, n2, n1, one, c.Off(i-1, n2), q.Off(n1, n2), one); err != nil {
					panic(err)
				}

				//              Copy everything back.
				Dlacpy(Full, len, n, work.Matrix(ldwork, opts), c.Off(i-1, 0))
			}
		}
	}

	work.Set(0, float64(lwkopt))

	return
}
