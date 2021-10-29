package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgglse solves the linear equality-constrained least squares (LSE)
// problem:
//
//         minimize || c - A*x ||_2   subject to   B*x = d
//
// where A is an M-by-N matrix, B is a P-by-N matrix, c is a given
// M-vector, and d is a given P-vector. It is assumed that
// P <= N <= M+P, and
//
//          rank(B) = P and  rank( (A) ) = N.
//                               ( (B) )
//
// These conditions ensure that the LSE problem has a unique solution,
// which is obtained using a generalized RQ factorization of the
// matrices (B, A) given by
//
//    B = (0 R)*Q,   A = Z*T*Q.
func Zgglse(m, n, p int, a, b *mat.CMatrix, c, d, x, work *mat.CVector, lwork int) (info int, err error) {
	var lquery bool
	var cone complex128
	var lopt, lwkmin, lwkopt, mn, nb, nb1, nb2, nb3, nb4, nr int

	cone = (1.0 + 0.0*1i)

	//     Test the input parameters
	mn = min(m, n)
	lquery = (lwork == -1)
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if p < 0 || p > n || p < n-m {
		err = fmt.Errorf("p < 0 || p > n || p < n-m: p=%v, m=%v, n=%v", p, m, n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if b.Rows < max(1, p) {
		err = fmt.Errorf("b.Rows < max(1, p): b.Rows=%v, p=%v", b.Rows, p)
	}

	//     Calculate workspace
	if err == nil {
		if n == 0 {
			lwkmin = 1
			lwkopt = 1
		} else {
			nb1 = Ilaenv(1, "Zgeqrf", []byte{' '}, m, n, -1, -1)
			nb2 = Ilaenv(1, "Zgerqf", []byte{' '}, m, n, -1, -1)
			nb3 = Ilaenv(1, "Zunmqr", []byte{' '}, m, n, p, -1)
			nb4 = Ilaenv(1, "Zunmrq", []byte{' '}, m, n, p, -1)
			nb = max(nb1, nb2, nb3, nb4)
			lwkmin = m + n + p
			lwkopt = p + mn + max(m, n)*nb
		}
		work.SetRe(0, float64(lwkopt))

		if lwork < lwkmin && !lquery {
			err = fmt.Errorf("lwork < lwkmin && !lquery: lwork=%v, lwkmin=%v, lquery=%v", lwork, lwkmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zgglse", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Compute the GRQ factorization of matrices B and A:
	//
	//            B*Q**H = (  0  T12 ) P   Z**H*A*Q**H = ( R11 R12 ) N-P
	//                        N-P  P                     (  0  R22 ) M+P-N
	//                                                      N-P  P
	//
	//     where T12 and R11 are upper triangular, and Q and Z are
	//     unitary.
	if err = Zggrqf(p, m, n, b, work, a, work.Off(p), work.Off(p+mn), lwork-p-mn); err != nil {
		panic(err)
	}
	lopt = int(work.GetRe(p + mn + 1 - 1))

	//     Update c = Z**H *c = ( c1 ) N-P
	//                       ( c2 ) M+P-N
	if err = Zunmqr(Left, ConjTrans, m, 1, mn, a, work.Off(p), c.CMatrix(max(1, m), opts), work.Off(p+mn), lwork-p-mn); err != nil {
		panic(err)
	}
	lopt = max(lopt, int(work.GetRe(p+mn)))

	//     Solve T12*x2 = d for x2
	if p > 0 {
		if info, err = Ztrtrs(Upper, NoTrans, NonUnit, p, 1, b.Off(0, n-p), d.CMatrix(p, opts)); err != nil {
			panic(err)
		}

		if info > 0 {
			info = 1
			return
		}

		//        Put the solution in X
		goblas.Zcopy(p, d.Off(0, 1), x.Off(n-p, 1))

		//        Update c1
		err = goblas.Zgemv(NoTrans, n-p, p, -cone, a.Off(0, n-p), d.Off(0, 1), cone, c.Off(0, 1))
	}
	//
	//     Solve R11*x1 = c1 for x1
	//
	if n > p {
		if info, err = Ztrtrs(Upper, NoTrans, NonUnit, n-p, 1, a, c.CMatrix(n-p, opts)); err != nil {
			panic(err)
		}

		if info > 0 {
			info = 2
			return
		}

		//        Put the solutions in X
		goblas.Zcopy(n-p, c.Off(0, 1), x.Off(0, 1))
	}

	//     Compute the residual vector:
	if m < n {
		nr = m + p - n
		if nr > 0 {
			if err = goblas.Zgemv(NoTrans, nr, n-m, -cone, a.Off(n-p, m), d.Off(nr, 1), cone, c.Off(n-p, 1)); err != nil {
				panic(err)
			}
		}
	} else {
		nr = p
	}
	if nr > 0 {
		if err = goblas.Ztrmv(Upper, NoTrans, NonUnit, nr, a.Off(n-p, n-p), d.Off(0, 1)); err != nil {
			panic(err)
		}
		goblas.Zaxpy(nr, -cone, d.Off(0, 1), c.Off(n-p, 1))
	}
	//
	//     Backward transformation x = Q**H*x
	//
	if err = Zunmrq(Left, ConjTrans, n, 1, p, b, work.Off(0), x.CMatrix(n, opts), work.Off(p+mn), lwork-p-mn); err != nil {
		panic(err)
	}
	work.SetRe(0, float64(p+mn+max(lopt, int(work.GetRe(p+mn)))))

	return
}
